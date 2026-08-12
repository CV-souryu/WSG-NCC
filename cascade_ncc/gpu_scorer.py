"""Cascade scoring in TWO WGSL dispatches: batch prune -> select (top-N + NCC).

The gallery (hist 3362x576, sparse samples/valid 3362x384, common mask) is
uploaded once and persists on the GPU (~17 MB). Each query batch uploads only
the 576-d histograms + sparse query gray/valid, then:

  1. ``prune_scores`` — batch-parallel: a 2D grid of one thread per gallery row
     computes score[j] = hist_gallery[j] . feats. High occupancy hides the
     6.9MB hist read latency (the old single-threadgroup-per-image version left
     ~99% of GPU bandwidth unused for a single query).
  2. ``select_topk`` — one threadgroup per image reads the pruned global
     scores, keeps the top-N, refines each candidate with the exact NCC,
     sorts, and writes top-k.

Eliminates the CPU prune dot + argpartition + NCC-refine section (~4 ms).
"""

from __future__ import annotations

import numpy as np

from ._constants import (
    HIST_DIM,
    K_DEFAULT,
    MAX_CANDIDATES,
    MAX_QUERIES_DEFAULT,
    TOP_N_DEFAULT,
)
from ._gpu import (
    GPU_LOCK,
    compile_module,
    create_buffer,
    default_device,
    download,
    enqueue,
    make_pipelines,
    require_gpu,
    upload,
)

PARAM_UINTS = 6         # nimg, ngallery, npts, hdim, k, topn

# One WGSL source, translated at runtime by naga to MSL / SPIR-V / HLSL.
# The byte-typed gallery/query arrays are u32 arrays holding one 0-255 value
# per code point (WGSL has no 8-bit type). MAX_CANDIDATES is substituted below
# to size the threadgroup arrays. Binding names are unique per module scope.
WGSL = r"""
@group(0) @binding(0) var<storage, read>      hist: array<f32>;
@group(0) @binding(1) var<storage, read>      feats: array<u32>;
@group(0) @binding(2) var<storage, read_write> pscores: array<f32>;
@group(0) @binding(3) var<storage, read>      pparams: array<u32>;

// Batch-parallel prune: one thread per gallery row, 2D dispatch
// (ceil(ng/256) x m). The 6.9MB hist read is spread across many warps so the
// memory latency is hidden — near full bandwidth instead of ~1% for one query.
@compute @workgroup_size(256)
fn prune_scores(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let img = gid.y;
    let ng = pparams[1]; let hdim = pparams[3];
    if (j >= ng || img >= pparams[0]) { return; }
    var s = 0.0;
    for (var d: u32 = 0u; d < hdim; d = d + 1u) {
        s = s + hist[j * hdim + d] * f32(feats[img * hdim + d]);
    }
    pscores[img * ng + j] = s;
}

@group(0) @binding(0) var<storage, read>      sscores: array<f32>;
@group(0) @binding(1) var<storage, read>      samples: array<u32>;
@group(0) @binding(2) var<storage, read>      valid: array<u32>;
@group(0) @binding(3) var<storage, read>      cmn: array<u32>;
@group(0) @binding(4) var<storage, read>      q8: array<u32>;
@group(0) @binding(5) var<storage, read>      qv8: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_scores: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_idx: array<u32>;
@group(0) @binding(8) var<storage, read>      sparams: array<u32>;

var<workgroup> cand: array<u32, MAX_CANDIDATES>;
var<workgroup> texact: array<f32, MAX_CANDIDATES>;
var<workgroup> tglidx: array<u32, MAX_CANDIDATES>;
// per-thread refine partial sums (256 threads x 6 accumulators) for the
// parallel exact-NCC reduction
var<workgroup> sh_partial: array<f32, 1536u>;

// One threadgroup per query image: read the pruned global scores (in L2 after
// prune_scores), thread 0 keeps the top-N, one thread per candidate refines
// with the exact NCC, thread 0 sorts + writes top-k.
@compute @workgroup_size(256)
fn select_topk(@builtin(workgroup_id) img_v: vec3<u32>,
               @builtin(local_invocation_id) ltid_v: vec3<u32>) {
    let img = img_v.x;
    let ltid = ltid_v.x;
    let ng = sparams[1]; let npts = sparams[2];
    let k = sparams[4]; let topn = sparams[5];
    // 1. top-N: thread 0 keeps the N best indices (reads global scores)
    if (ltid == 0u) {
        var top_idx: array<u32, MAX_CANDIDATES>;
        var top_val: array<f32, MAX_CANDIDATES>;
        var n: u32 = 0u; var mi: u32 = 0u;
        for (var j: u32 = 0u; j < ng; j = j + 1u) {
            let s = sscores[img * ng + j];
            if (n < topn) {
                top_idx[n] = j; top_val[n] = s; n = n + 1u;
                if (s < top_val[mi]) { mi = n - 1u; }      // running min index
            } else if (s > top_val[mi]) {
                top_idx[mi] = j; top_val[mi] = s;          // replace, re-scan min
                mi = 0u;
                for (var t: u32 = 1u; t < topn; t = t + 1u) {
                    if (top_val[t] < top_val[mi]) { mi = t; }
                }
            }
        }
        for (var t: u32 = 0u; t < topn; t = t + 1u) { cand[t] = top_idx[t]; }
    }
    workgroupBarrier();
    // 2. refine: exact per-common-point NCC, PARALLELIZED across all 256
    // threads. tc = 256/topn threads per candidate each sum a slice of the
    // npts points into workgroup memory; thread (c,0) reduces and writes the
    // exact score. The serial fallback covers topn > 256. (``topn <= 256`` is
    // uniform, so the workgroupBarriers are uniform in either branch.)
    if (topn <= 256u) {
        let tc = 256u / topn;
        let c = ltid / tc;
        let tl = ltid % tc;
        if (c < topn) {
            let gidx = cand[c];
            var cnt = 0.0; var sg = 0.0; var sq = 0.0; var sg2 = 0.0; var sq2 = 0.0; var sgq = 0.0;
            for (var pp: u32 = tl; pp < npts; pp = pp + tc) {
                if (cmn[pp] != 0u && valid[gidx * npts + pp] != 0u
                    && qv8[img * npts + pp] != 0u) {
                    let gp = f32(samples[gidx * npts + pp]);
                    let qp = f32(q8[img * npts + pp]);
                    cnt = cnt + 1.0; sg = sg + gp; sq = sq + qp;
                    sg2 = sg2 + gp * gp; sq2 = sq2 + qp * qp; sgq = sgq + gp * qp;
                }
            }
            sh_partial[ltid * 6u + 0u] = cnt;
            sh_partial[ltid * 6u + 1u] = sg;
            sh_partial[ltid * 6u + 2u] = sq;
            sh_partial[ltid * 6u + 3u] = sg2;
            sh_partial[ltid * 6u + 4u] = sq2;
            sh_partial[ltid * 6u + 5u] = sgq;
        }
        workgroupBarrier();
        if (c < topn && tl == 0u) {
            var rcnt = 0.0; var rsg = 0.0; var rsq = 0.0; var rsg2 = 0.0; var rsq2 = 0.0; var rsgq = 0.0;
            for (var i: u32 = 0u; i < tc; i = i + 1u) {
                let b = (c * tc + i) * 6u;
                rcnt = rcnt + sh_partial[b + 0u]; rsg = rsg + sh_partial[b + 1u];
                rsq = rsq + sh_partial[b + 2u]; rsg2 = rsg2 + sh_partial[b + 3u];
                rsq2 = rsq2 + sh_partial[b + 4u]; rsgq = rsgq + sh_partial[b + 5u];
            }
            var ex = -1.0;
            if (rcnt >= 50.0) {
                let mg = rsg / rcnt; let mq = rsq / rcnt;
                let vg = max(rsg2 / rcnt - mg * mg, 0.0);
                let vq = max(rsq2 / rcnt - mq * mq, 0.0);
                let cv = rsgq / rcnt - mg * mq;
                ex = cv / sqrt(max(vg * vq, 1e-12));
            }
            texact[c] = ex;
            tglidx[c] = cand[c];
        }
        workgroupBarrier();
    } else {
        if (ltid < topn) {
            let c = cand[ltid];
            var cnt = 0.0; var sg = 0.0; var sq = 0.0; var sg2 = 0.0; var sq2 = 0.0; var sgq = 0.0;
            for (var pp: u32 = 0u; pp < npts; pp = pp + 1u) {
                if (cmn[pp] != 0u && valid[c * npts + pp] != 0u
                    && qv8[img * npts + pp] != 0u) {
                    let gp = f32(samples[c * npts + pp]);
                    let qp = f32(q8[img * npts + pp]);
                    cnt = cnt + 1.0; sg = sg + gp; sq = sq + qp;
                    sg2 = sg2 + gp * gp; sq2 = sq2 + qp * qp; sgq = sgq + gp * qp;
                }
            }
            var ex = -1.0;
            if (cnt >= 50.0) {
                let mg = sg / cnt; let mq = sq / cnt;
                let vg = max(sg2 / cnt - mg * mg, 0.0);
                let vq = max(sq2 / cnt - mq * mq, 0.0);
                let cv = sgq / cnt - mg * mq;
                ex = cv / sqrt(max(vg * vq, 1e-12));
            }
            texact[ltid] = ex;
            tglidx[ltid] = c;
        }
        workgroupBarrier();
    }
    // 3. output: sort top-N by exact, write top-k
    if (ltid == 0u) {
        for (var i: u32 = 0u; i < topn; i = i + 1u) {
            for (var j: u32 = i + 1u; j < topn; j = j + 1u) {
                if (texact[j] > texact[i]) {
                    let e = texact[i]; texact[i] = texact[j]; texact[j] = e;
                    let g = tglidx[i]; tglidx[i] = tglidx[j]; tglidx[j] = g;
                }
            }
        }
        let kk = min(k, topn);
        for (var i: u32 = 0u; i < kk; i = i + 1u) {
            out_scores[img * k + i] = texact[i];
            out_idx[img * k + i] = tglidx[i];
        }
    }
}
""".replace("MAX_CANDIDATES", str(MAX_CANDIDATES))   # sizes the threadgroup score arrays


class CascadeGpuScorer:
    """Batch prune + select (top-N + exact NCC) for the cascade."""

    def __init__(self, cb, max_queries: int = MAX_QUERIES_DEFAULT,
                 device=None):
        require_gpu("GPU scorer")
        ng = len(cb.paths)
        npts = len(cb.common8)
        # npts (sparse code-point count) is NOT bounded by the kernel — the refine
        # loop walks it dynamically. Only topn/k are capped by MAX_CANDIDATES.
        self.ngallery = ng
        self.cb = cb
        self.max_queries = max_queries
        self.device = device or default_device()
        module = compile_module(self.device, WGSL, "GPU scorer")
        self.pipelines = make_pipelines(
            self.device, module,
            {"prune_scores": ["read-only-storage", "read-only-storage",
                              "storage", "read-only-storage"],
             "select_topk": ["read-only-storage"] * 6
                            + ["storage", "storage"] + ["read-only-storage"]},
            "GPU scorer")
        self.prune_pipe, self.prune_bgl = self.pipelines["prune_scores"]
        self.select_pipe, self.select_bgl = self.pipelines["select_topk"]
        self.hist_buf = create_buffer(self.device, ng * HIST_DIM * 4)
        upload(self.device, self.hist_buf,
              np.ascontiguousarray(cb.hist, np.float32).ravel())
        self.samples_buf = create_buffer(self.device, ng * npts * 4)
        upload(self.device, self.samples_buf,
              np.ascontiguousarray(cb.samples8, np.uint8).ravel().astype(np.uint32))
        self.valid_buf = create_buffer(self.device, ng * npts * 4)
        upload(self.device, self.valid_buf,
              np.ascontiguousarray(cb.valid8, np.uint8).ravel().astype(np.uint32))
        self.common_buf = create_buffer(self.device, npts * 4)
        upload(self.device, self.common_buf,
              np.asarray(cb.common8, np.uint8).ravel().astype(np.uint32))
        # Query working buffers (feats/q8/qv8) are NOT resident: the fused path
        # reads them from the sampler's GPU buffers, and the standalone score()
        # allocates them locally. Only the result + prune buffers persist.
        # Single result buffer: [scores (f32) x max_queries*MAX_CANDIDATES]
        # then [idx (u32) x max_queries*MAX_CANDIDATES], read back in ONE call
        # (two wgpu read_buffer calls each pay a fixed ~1.4ms map_sync latency).
        # The offset is always 256-aligned (MAX_CANDIDATES*4 is a multiple of 256).
        self._seg = max_queries * MAX_CANDIDATES * 4   # scores segment (bytes)
        self.out_buf = create_buffer(self.device, self._seg * 2)
        self.scores_buf = create_buffer(
            self.device, max_queries * ng * 4)
        self.p_buf = create_buffer(self.device, PARAM_UINTS * 4)

    def score(self, feats: np.ndarray, q8: np.ndarray, qv8: np.ndarray,
              k: int = K_DEFAULT, top_n: int = TOP_N_DEFAULT):
        """Two dispatches; returns (idx, scores) arrays of shape (M, k).

        Serialized by the global GPU lock (the shared device + working buffers
        are not thread-safe).
        """
        with GPU_LOCK:
            return self._score_locked(feats, q8, qv8, k, top_n)

    def _score_locked(self, feats: np.ndarray, q8: np.ndarray, qv8: np.ndarray,
                      k: int = K_DEFAULT, top_n: int = TOP_N_DEFAULT):
        m = feats.shape[0]
        if m > self.max_queries:
            raise ValueError(f"{m} queries exceed max_queries {self.max_queries}")
        if k > MAX_CANDIDATES or top_n > MAX_CANDIDATES:
            raise ValueError(
                f"k={k} / top_n={top_n} exceed MAX_CANDIDATES {MAX_CANDIDATES}")
        npts = len(self.cb.common8)
        ng = len(self.cb.paths)
        # Standalone-only: the query buffers are transient here (the fused path
        # reads them from the sampler's GPU buffers instead), so allocate them
        # locally and let them be GC'd after the readback.
        feats_buf = create_buffer(self.device, m * HIST_DIM * 4)
        q8_buf = create_buffer(self.device, m * npts * 4)
        qv8_buf = create_buffer(self.device, m * npts * 4)
        upload(self.device, feats_buf,
              np.ascontiguousarray(feats, np.uint32).ravel())
        upload(self.device, q8_buf,
              np.ascontiguousarray(q8, np.uint8).ravel().astype(np.uint32))
        upload(self.device, qv8_buf,
              np.ascontiguousarray(qv8, np.uint8).ravel().astype(np.uint32))
        upload(self.device, self.p_buf,
               np.asarray([m, ng, npts, HIST_DIM, k, top_n], np.uint32))
        encoder = self.device.create_command_encoder()
        p1 = encoder.begin_compute_pass()
        enqueue(p1, self.device, self.prune_pipe, self.prune_bgl,
                [self.hist_buf, feats_buf, self.scores_buf, self.p_buf],
                ((ng + 255) // 256, m))
        p1.end()
        p2 = encoder.begin_compute_pass()
        enqueue(p2, self.device, self.select_pipe, self.select_bgl,
                [self.scores_buf, self.samples_buf, self.valid_buf,
                 self.common_buf, q8_buf, qv8_buf,
                 (self.out_buf, 0, self._seg),
                 (self.out_buf, self._seg, self._seg), self.p_buf], m)
        p2.end()
        self.device.queue.submit([encoder.finish()])
        return self.read_topk(m, k)

    def enqueue_prune(self, pass_, sd, m, k, top_n):
        """Record the batch-parallel prune into a compute pass (2D dispatch).

        Reads the query histogram straight from the sampler's ``sd.hist_buf``
        (no CPU round-trip) and writes the raw prune scores to ``scores_buf``.
        The prune ranks RAW histogram counts — a constant per-image scale does
        not change the top-N order.
        """
        if k > MAX_CANDIDATES or top_n > MAX_CANDIDATES:
            raise ValueError(
                f"k={k} / top_n={top_n} exceed MAX_CANDIDATES {MAX_CANDIDATES}")
        npts = len(self.cb.common8)
        upload(self.device, self.p_buf,
               np.asarray([m, self.ngallery, npts, HIST_DIM, k, top_n], np.uint32))
        enqueue(pass_, self.device, self.prune_pipe, self.prune_bgl,
                [self.hist_buf, sd.hist_buf, self.scores_buf, self.p_buf],
                ((self.ngallery + 255) // 256, m))

    def enqueue_select(self, pass_, sd, m, k, top_n):
        """Record the top-N + exact-NCC select into a compute pass.

        Must run after ``enqueue_prune`` (same command buffer, later pass) so
        ``scores_buf`` holds the pruned scores for this image.
        """
        if k > MAX_CANDIDATES or top_n > MAX_CANDIDATES:
            raise ValueError(
                f"k={k} / top_n={top_n} exceed MAX_CANDIDATES {MAX_CANDIDATES}")
        npts = len(self.cb.common8)
        upload(self.device, self.p_buf,
               np.asarray([m, self.ngallery, npts, HIST_DIM, k, top_n], np.uint32))
        enqueue(pass_, self.device, self.select_pipe, self.select_bgl,
                [self.scores_buf, self.samples_buf,
                 self.valid_buf, self.common_buf,
                 sd.sa_srgb_buf, sd.sa_svalid_buf,
                 (self.out_buf, 0, self._seg),
                 (self.out_buf, self._seg, self._seg), self.p_buf], m)

    def read_topk(self, m, k):
        """Read the combined [scores | idx] buffer in ONE readback and split."""
        raw = np.frombuffer(download(self.device, self.out_buf, self._seg * 2),
                            np.uint8).copy()
        scores = raw[: self._seg].view(np.float32)[: m * k].copy().reshape(m, k)
        idx = raw[self._seg:].view(np.uint32)[: m * k].copy().reshape(m, k)
        return idx, scores
