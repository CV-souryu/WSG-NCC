# Cascade NCC（船卡识别）

单工件级联识别：**512 维全局颜色直方图粗筛 + 稀疏 9x9 公共码点精确 NCC**，
全部打包进一个 `.npz` 码本。批量识别走 wgpu GPU（一份 WGSL shader 由 wgpu-native
自动翻译到 Vulkan / Metal / DX12 并缓存编译产物；码本常驻 GPU、一次 dispatch），
无 GPU 时自动回退 CPU。spiral / concentric / level 旧路径已全部删除，仅保留
cascade 一条链路。

## 安装

```bash
python3 -m venv .venv
.venv/bin/pip install -e .            # 核心（CPU）
.venv/bin/pip install -e ".[gpu]"     # 跨平台 GPU 加速（wgpu，Vulkan / Metal / DX12）
.venv/bin/pip install -e ".[test]"    # 测试 / lint 依赖（pytest、ruff）
```

> GPU 依赖 `wgpu`（wgpu-native 自动选择 Vulkan / Metal / DX12 后端，运行时把一份
> WGSL 翻译成各后端 shader 并缓存）。无 GPU 时 `CascadeShipRecognizer` 会打印提示
> 并回退 CPU，功能不受影响。

## 命令行

```bash
# 单张 / 多张识别（默认 codebook=cascade，backend=gpu）
.venv/bin/cascade-ncc recognize data/groups/group1/testset/cards/001.png --k 3
.venv/bin/cascade-ncc recognize data/groups/group1/testset/cards/001.png \
    data/groups/group1/testset/cards/002.png --codebook cascade --k 3 --backend cpu

# 等价于：python -m cascade_ncc.cli recognize ...
```

输出每张图一行文件名，下面按名次列出 top-k（gallery 路径 + NCC 分数）：

```
data/groups/group1/testset/cards/001.png
  1.	.../data/groups/group1/gallery/1/1426/XM_NORMAL_1426.png	0.9390
  2.	.../data/groups/group1/gallery/1/1006/XM_NORMAL_1006.png	0.5887
  3.	.../data/groups/group1/gallery/2B/6_2/XM_BROKEN_6_2.png	0.5412
```

参数：`--codebook`（名字解析到 `data/codebooks/<name>.npz`，或直接给 `.npz` 路径）、
`--k`（每图 top-k）、`--backend`（`cpu` / `gpu`，gpu 不可用时自动回退）。

## 库 API

```python
from cascade_ncc import (CascadeCodebook, CascadeShipRecognizer,
                         build_cascade_codebook, load_cascade_codebook,
                         recognize_cascade)

# 建库（自动缓存到 data/codebooks/<name>.npz）
cb = build_cascade_codebook(gallery_paths, name="cascade")   # list[Path]

# 加载
cb = load_cascade_codebook("cascade")              # 按名字
cb = load_cascade_codebook("data/codebooks/cascade.npz")  # 按路径

# 函数式识别：返回 [(gallery_index, Path, score), ...]
top = recognize_cascade(cb, "query.png", k=5, top_n=20)

# 类接口：GPU 批量（默认）或 CPU
r = CascadeShipRecognizer("cascade", use_gpu=True)     # GPU 批量（wgpu），自动回退 CPU
top = r.recognize(img_rgba_u8, k=3)                    # 单图 -> 一个结果列表
tops = r.recognize([img1, img2, ...], k=3)             # 批量 -> 每图一个列表
```

- **查询输入**：文件路径，或 `(H, W, 3/4)` 的 uint8 numpy 数组（RGB 或 RGBA），
  单图与批量走完全相同的预处理。
- **码本参数**（`build_cascade_codebook`）：`step=2` 稠密网格驱动直方图、
  `ncc_step=8` 稀疏子集 + `ncc_pool=9` 像素邻域做精确 NCC、
  `bins=8`（→512 维直方图）、`top_fraction=0.8`（丢弃底部 20% 高方差区域）。
- **画布可参数化**：`cw`/`ch`（默认 `124 × 240`）决定码点的采样画布，会写进码本
  `params` 并进入缓存 key；CPU 预处理与 GPU 内核都按码本里的画布 resize。用
  不同画布建的码本会得到不同的缓存 key，不会误用旧缓存。
- **预处理配置**（`trim_blue` / `shift_y` / `align`，都会写进码本 params 并进入
  缓存 key）：
  - `trim_blue=True`：先裁掉蓝色边框（`B > R+20 && B > G+20` 的包围盒）。
  - `align="<垂直>-<水平>"`：内容贴合的边，垂直 ∈ `top/center/bottom` × 水平 ∈
    `left/center/right`（默认 `"top-center"`）。cover 缩放固定；溢出在非贴合侧
    **裁切**，像素不足（含 shift_y 边距）用**透明黑填充**。
  - `shift_y`：附加垂直偏移（默认 4，顶部留黑边）。
- `CascadeShipRecognizer` 的 `trim_blue`/`shift_y`/`align` **默认从码本 params
  读取**（旧码本缺省回退到 `True`/`4`/`"top-center"`）——识别自动匹配码本记录
  的预处理，避免配置不一致导致弱匹配。显式传参会覆盖码本值。

## GPU 后端（wgpu）

- **单源 shader**：全部内核是一份 WGSL，wgpu-native 运行时按后端翻译成
  MSL（Metal）/ SPIR-V（Vulkan）/ HLSL（DX12）并缓存编译产物。Python host 代码
  后端无关。
- **近似实现**：GPU 用双线性 + f32 几何，CPU 用 LANCZOS + f64——top-1 与 CPU
  一致（58/58 测试卡），但 top-3 里近并列条目的**顺序**可能与 CPU 有
  ~0.0001 级分数的翻转，属预期。
- **线程模型**：GPU 工作集（共享 device + 共享 preprocess + 各 stage 缓冲）
  非线程安全，所有 GPU 推理由**一把全局锁串行化**。多线程/多 recognizer 调用
  安全，但 GPU 本就单设备单队列，串行不损失真实吞吐。
- **内存**：多码本共享一个 device + 一份查询工作集（~30 MB），每个码本额外
  常驻 gallery ~20 MB。N 码本 ≈ `N×20 + 30` MB。
- **性能**（单机实测）：单张 GPU ~1.7ms vs CPU ~4.2ms；批量 58 GPU ~13ms vs
  CPU ~150ms。批量越大 GPU 优势越大（固定开销摊销）。

## 数据组织

数据集按组归并，每组一个自洽的原图库 + 测试集（`data/groups/`，约定见
`data/groups/README.md`）：

```
data/
├── groups/
│   ├── group1/
│   │   ├── gallery/        原图库（1/ 2/ 2B/，共 3362 张）
│   │   └── testset/        测试集（cards/ + screens/ + alignment.json + summary.json）
│   └── group2/             预留第二组
├── ship_names.json         全局船名映射
└── fonts/                  生成用素材（gitignored）
```

**新增一组**：建 `data/groups/group2/{gallery,testset}`，放入数据后用
`build_cascade_codebook(sorted(Path("data/groups/group2/gallery").rglob("*.png")),
name="cascade-group2")` 重建码本。注意码本记录的是**绝对路径**，gallery 移动或
换组后必须重建（默认 `cascade.npz` 也一样）。

## 目录

```
├── cascade_ncc/                 # 核心库
│   ├── codebook.py              # 码本数据类 + 几何/构建/加载/缓存
│   ├── recognizer.py            # recognize_cascade + CascadeShipRecognizer（GPU 编排）
│   ├── codebook_match.py        # 精确 NCC 打分
│   ├── primitives.py            # 采样/缩放/预处理底层
│   ├── gpu_preprocess.py        # GPU 融合预处理（bbox + cover-resize + shift）
│   ├── gpu_sampler.py           # GPU 融合采样（稠密直方图 + 稀疏 NCC）
│   ├── gpu_scorer.py            # GPU 融合打分（prune + top-N + 精确 NCC）
│   ├── _gpu.py                  # 共享 wgpu 样板（设备/模块/管线/绑定/派发）
│   ├── _constants.py            # 共享常量（画布尺寸、阈值、灰度权重）
│   └── cli.py                   # cascade-ncc 命令行入口
├── data/
│   ├── groups/group1/           # 原图库 + 测试集（见「数据组织」）
│   └── codebooks/               # 命名码本 .npz（cascade.npz 等，gitignored）
├── tests/
│   ├── conftest.py              # 夹具 + 共享 helper（合成卡图）
│   ├── test_cpu.py              # CPU 底层 primitives 测试（无 GPU/数据依赖）
│   ├── test_codebook.py         # 码本构建/识别/CPU 批量测试（合成图）
│   └── test_gpu.py              # GPU kernel 正确性（需 wgpu + 数据，否则 skip）
├── docs/
│   └── archive/                 # 历史实验记录（SPEED.md、bluecard，已 OUTDATED）
└── pyproject.toml
```

## 测试与 lint

```bash
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest                     # 全量（CPU + GPU）
.venv/bin/python -m pytest tests/test_cpu.py tests/test_codebook.py   # 仅 CPU
.venv/bin/python -m ruff check cascade_ncc tests
```

GPU 测试需要 `wgpu`（Vulkan / Metal / DX12；无真实 GPU 时可装 lavapipe 软件
Vulkan）+ `data/groups/group1/testset/`；缺失时自动 skip，CPU 测试在任何环境都能跑。
