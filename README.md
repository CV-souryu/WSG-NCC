# Cascade NCC（船卡识别）

单工件级联识别：**576 维 H16S2L2 × 3×3 空间直方图粗筛 + 稀疏 9x9 公共码点精确 NCC**，
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
`--k`（每图 top-k）、`--backend`（`cpu` / `gpu`，gpu 不可用时自动回退）、
`--min-confidence`（低于该 top-1 分的不输出，默认 0.4；设 0 关闭过滤）、
`--fit-width / --no-fit-width`（宽缩放 / cover，不传用码本 params）、
`--unmask`（除以该因子还原被遮罩压暗的 RGB，0 显式关闭，不传用码本 params）、
`--region TOP BOTTOM LEFT RIGHT`（只激活中心落在该百分比区域内的 3×3 直方图
分区，如 `--region 0 50 0 100` 激活上方 6 个分区）。

## 推荐参数（实测调优）

针对 live-target-selector 裁出的单卡（`card-XX-rYcZ.png` 形式），以下两套参数
在 14 张实测卡上全部命中（top-1 正确：B 组 9/9、A 组 5/5）：

| 场景 | trim_blue | shift_y | unmask | region |
|---|---|---:|---:|---|
| 默认（正常亮度） | 切蓝边 | 4 | 0 | `(0, 60, 0, 100)` 激活顶部 60% |
| 蒙版状态（半透明黑罩压暗） | 切蓝边 | 4 | 0.33 | `(0, 40, 0, 100)` 激活顶部 40% |

- 默认配置用于正常画面：只激活画布顶部 60% 的码点，避开底部高方差区域。
- 蒙版配置用于整体被约 0.33 不透明度黑罩压暗的画面：先用 `unmask=0.33` 还原
  RGB，再只激活顶部 40% 码点。
- 两套都直接用全码本（`cascade.npz`，`top_fraction=1.0`），通过 `region` 在
  推理时激活码点，**不需要**为不同比例重建码本。
- trim 蓝边阈值已调严到 `B > R+80 && B > G+80`（`BLUE_TRIM_THRESH`）：真实蓝边
  是固定色（约 RGB 15/125/214），该阈值既能裁掉边框，又不会误裁卡面顶部的
  淡蓝色区域。
- **二次确认**：默认参数（正常亮度）无结果或分数偏低时，改用蒙版参数
  （`unmask=0.33` + 顶部 40%）再跑一次；若蒙版参数给出高分结果，说明该图实际
  处于蒙版状态，以蒙版结果为准。

CLI：

```bash
# 默认：切蓝边 + sy4 + 0mask + 顶部60%
.venv/bin/cascade-ncc recognize card.png --k 3 --region 0 60 0 100

# 蒙版：切蓝边 + sy4 + unmask0.33 + 顶部40%
.venv/bin/cascade-ncc recognize card.png --k 3 --unmask 0.33 --region 0 40 0 100
```

库 API：

```python
rec = CascadeRecognizer("cascade", region=(0, 60, 0, 100))               # 默认
rec = CascadeRecognizer("cascade", region=(0, 40, 0, 100), unmask=0.33)  # 蒙版
```

### 码本构建推荐参数

码本一律建**全码本**（`top_fraction=1.0`），上 60% / 40% 的激活全部留给推理时的
`region` 参数——一组码本即可服务上面的两套推理配置，不需要为不同比例重复建码本：

```python
from pathlib import Path
from cascade_ncc import build_cascade_codebook

gallery = sorted(Path("data/groups/group1/gallery").rglob("*.png"))
build_cascade_codebook(
    gallery,
    step=2, ncc_step=8, ncc_pool=9,       # 稠密 step=2 + 稀疏 step=8 + 9×9 池化
    hue_bins=16, sat_bins=2, lig_bins=2,  # 576 维 H16S2L2 × 3×3 直方图
    cells=(3, 3),
    min_common_frac=0.9,                  # 公共码点：≥90% gallery 有效
    top_fraction=1.0,                     # 全码本，上部分激活由推理 region 控制
    cw=124, ch=240,
    trim_blue=True, shift_y=4,            # 记录进 params，识别自动读取
    align="top-center", fit_width=True,
    unmask=0.0,                           # gallery 原图本身无蒙版
    name="cascade",
)
```

注意：`trim_blue` / `shift_y` / `align` / `fit_width` / `unmask` 会写进码本
`params`，识别器默认按码本记录值预处理；推理时用 `--unmask` / `--region` 等参数
覆盖即可，不需要为“默认 / 蒙版”两种场景各建一份码本。

## 库 API

```python
from cascade_ncc import (CascadeCodebook, CascadeRecognizer,
                         CascadeShipRecognizer, build_cascade_codebook,
                         load_cascade_codebook, recognize_cascade)
```

### 函数式接口

`build_cascade_codebook(paths, step=2, ncc_step=8, ncc_pool=9,
hue_bins=16, sat_bins=2, lig_bins=2, cells=(3,3), min_common_frac=0.9,
top_fraction=0.8, cw=124, ch=240, trim_blue=True,
shift_y=4, align="top-center", name=None, cache_path=None, force=False)`
从 gallery 图片列表构建码本，返回 `CascadeCodebook`。给 `name` 会自动缓存到
`data/codebooks/<name>.npz`；`cache_path` 可指定其它路径；`force=True` 强制重建。

> 直方图分箱固定为 **H16S2L2 × 3×3**（`hue_bins=16`、`sat_bins=2`、
> `lig_bins=2`、`cells=(3,3)`）：WGSL GPU 内核硬编码这些维度，传其它值会在
> 构建时报错，避免 GPU 路径静默算错。

> **构建不缩放**：码本构建要求所有 gallery 图与画布 `cw × ch`（默认 `124 × 240`）
> 完全同尺寸，否则直接报错——长宽比不一的图请先统一裁切/预处理到画布尺寸，
> 而不是靠构建时的隐式拉伸（拉伸会让码点与查询预处理后的画布对不齐）。

`load_cascade_codebook(name_or_path)` 按名字（如 `"cascade"`）、`.npz` 路径或
`.npz` 原始 bytes 加载码本。

`recognize_cascade(cb, query, k=3, top_n=20, trim_blue=None, shift_y=None,
refine=None, align=None, fit_width=None, unmask=None, region=None)`
纯 CPU 单图识别，返回
`[(gallery_index, Path, score), ...]`，按分数从高到低。
预处理参数传 `None` 时从码本 `params` 读取；`refine=None` 精确打分整个
`top_n` 候选集（与 GPU 路径一致）。

### 公开类

#### CascadeCodebook（码本数据类）

不直接构造，由 `build_cascade_codebook` / `load_cascade_codebook` 返回。常用字段：

- `paths: list[Path]`——gallery 绝对路径，按码本内顺序。
- `hist: np.ndarray`——每张 gallery 的 576 维 H16S2L2 × 3×3 空间直方图。
- `samples8` / `valid8` / `common8` / `normed8`——稀疏 NCC 精确打分所需数据。
- `params: dict`——构建参数（`step` / `ncc_step` / `ncc_pool` /
  `hue_bins` / `sat_bins` / `lig_bins` / `cells` /
  `cw` / `ch` / `trim_blue` / `shift_y` / `align` 等），识别器会从这里读默认值。

```python
cb = build_cascade_codebook(gallery_paths, name="cascade")   # list[Path]
cb = load_cascade_codebook("cascade")                        # 按名字
cb = load_cascade_codebook("data/codebooks/cascade.npz")     # 按路径
top = recognize_cascade(cb, "query.png", k=5, top_n=20)      # 直接喂函数
```

#### CascadeShipRecognizer（低层识别器：GPU 批量 / CPU）

```python
CascadeShipRecognizer(
    codebook="cascade",        # 名字 / .npz 路径 / bytes / 已加载的 CascadeCodebook
    use_gpu=True,              # wgpu GPU 批量；不可用时自动回退 CPU
    max_queries=128,           # 单批上限，超出自动分批
    trim_blue=None,            # None = 从码本 params 读取
    shift_y=None,              # None = 从码本 params 读取
    top_n=20,                  # 粗筛候选数
    align=None,                # None = 从码本 params 读取
    fit_width=None,            # None = 从码本 params 读取
    unmask=None,               # None = 从码本 params 读取；0.0 显式关闭
    region=None,               # (top, bottom, left, right) % 区域激活直方图分区
    min_confidence=0.4,        # top-1 低于该分返回空列表；None 关闭过滤
)
```

`recognize(images, k=3)` 接受单张或批量输入（文件路径 / `(H, W, 3/4)` uint8
数组），返回：

- 单图输入：`[(gallery_index: int, gallery_path: Path, score: float), ...]`
- 批量输入（`list` / `tuple`）：上面这个列表的列表，每张图一个。

`recognize(images, k=3, min_confidence=0.4)`：低于阈值的匹配被丢弃，top-1 低于
阈值时该图返回空列表 `[]`。`min_confidence` 不传时用构造函数的值，构造函数传
`None` 关闭过滤（也可用 `min_confidence=0.0`）。

`recognize(images, k=3, fit_width=False, unmask=0.4, region=(0,40,0,100))`
支持**本次调用临时覆盖**预处理/激活配置（不改变实例本身）；内部按配置缓存
临时识别器，重复使用同一覆盖配置不会重复重建。

```python
r = CascadeShipRecognizer("cascade", use_gpu=True)
top = r.recognize(img_rgba_u8, k=3)          # 单图 -> 一个结果列表
tops = r.recognize([img1, img2, ...], k=3)   # 批量 -> 每图一个结果列表
```

实例属性：`cb`（码本）、`trim_blue` / `shift_y` / `align` / `fit_width` /
`unmask`（生效的预处理配置）、`top_n`、`use_gpu`、`max_queries`、
`min_confidence`（阈值过滤）。

#### CascadeRecognizer（高层识别器：码本 + 元数据）

在 `CascadeShipRecognizer` 之上加一层“值映射”：每个匹配返回
`(value, confidence, key)`，其中 `key` 是匹配到的 gallery 路径相对构建目录的
路径（如 `"1/226/XM_NORMAL_226.png"`），`value` 是 `meta[key]`（没有元数据时
为 `None`）。

```python
rec = CascadeRecognizer(
    codebook,                  # 名字 / .npz 路径 / bytes
    meta=None,                 # dict[key -> 任意值]，key 用相对路径
    k=3,                       # 默认 top-k
    use_gpu=True,
    max_queries=128,
    trim_blue=None,            # None = 从码本 params 读取
    shift_y=None,
    align=None,
    fit_width=None,            # None = 从码本 params 读取
    unmask=None,               # None = 从码本 params 读取；0.0 显式关闭
    region=None,               # (top, bottom, left, right) % 区域激活直方图分区
    min_confidence=0.4,        # top-1 低于该分返回空列表；None 关闭过滤
)
```

```python
rec = CascadeRecognizer(codebook_path_or_bytes)
meta = {key: key.split("/")[-1] for key in rec.keys}   # key → 自定义泛型值
rec = CascadeRecognizer(codebook_path_or_bytes, meta=meta)
top = rec.recognize(img_rgba_u8, k=3)   # [(值或None, 置信度, key), ...]
tops = rec.recognize([img1, img2], k=3) # 批量 -> 每图一个 [(值, 置信度, key), ...]
```

`rec.recognize(images, k=3, min_confidence=0.4)` 同样支持阈值过滤：低于 0.4 的
匹配被丢弃，top-1 低于阈值时返回空列表。不传时用构造函数的值，构造函数传
`None` 关闭过滤。

属性：`paths`（gallery 绝对路径，码本顺序）、`keys`（对应的相对路径）。
完整元数据（key → `{shipIndex, title}`）可由 `ship_names.json` 生成。

#### 示例：识别船仓舰船卡片

船仓截图先按卡片位置裁成单卡（本项目 `testset/cards/` 就是裁好的单卡），然后
交给 `CascadeRecognizer.recognize`：

```python
import json
from pathlib import Path

from cascade_ncc import CascadeRecognizer

# key（gallery 相对路径）→ 船名
meta = {
    key: info["title"]
    for key, info in json.loads(
        Path("data/gallery_meta.json").read_text(encoding="utf-8")
    ).items()
}

rec = CascadeRecognizer("cascade", meta=meta, k=3)   # GPU，不可用时自动回退 CPU

# 单张船卡
top = rec.recognize("data/groups/group1/testset/cards/001.png")
for ship, score, key in top:
    print(f"{ship}  {score:.4f}  {key}")

# 整个船仓的船卡批量识别
cards = sorted(Path("data/groups/group1/testset/cards").glob("*.png"))
for result in rec.recognize(cards, k=1):
    ship, score, key = result[0]
    print(f"{key}  ->  {ship}  {score:.4f}")
```

### 输入约定

- **查询输入**：文件路径，或 `(H, W, 3/4)` 的 uint8 numpy 数组（RGB 或 RGBA），
  单图与批量走完全相同的预处理。
- **码本参数**（`build_cascade_codebook`）：`step=2` 稠密网格驱动直方图、
  `ncc_step=8` 稀疏子集 + `ncc_pool=9` 像素邻域做精确 NCC、
  `hue_bins=16` / `sat_bins=2` / `lig_bins=2` × `cells=(3,3)`
  （→576 维空间直方图）、`top_fraction=0.8`（丢弃底部 20% 高方差区域）。
- **区域激活**：`region=(top, bottom, left, right)`（0–100，按画布百分比）只保留
  中心落在该区域内的 3×3 分区桶；如 `(0, 50, 0, 100)` 激活上方 2 行共 6 个分区，
  CPU/GPU 的直方图粗筛和稀疏 NCC 都只使用激活分区。
- **画布可参数化**：`cw`/`ch`（默认 `124 × 240`）决定码点的采样画布，会写进码本
  `params` 并进入缓存 key；CPU 预处理与 GPU 内核都按码本里的画布 resize。用
  不同画布建的码本会得到不同的缓存 key，不会误用旧缓存。
- **预处理配置**（`trim_blue` / `shift_y` / `align`，都会写进码本 params 并进入
  缓存 key）：
  - `trim_blue=True`：先裁掉蓝色边框（`B > R+80 && B > G+80` 的包围盒，
    `BLUE_TRIM_THRESH=80`，见「推荐参数」）。
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

**`data/` 整个 gitignore**——图片、码本、船名映射、测试集注解都随数据分发，不在仓库里。
数据依赖的测试在缺数据时自动 skip。磁盘上的布局（约定见 `data/groups/README.md`）：

```
data/
├── groups/
│   ├── group1/
│   │   ├── gallery/        原图库（1/ 2/ 2B/，共 3362 张）
│   │   └── testset/        测试集（cards/ + screens/ + alignment.json + summary.json）
│   └── group2/             预留第二组
├── codebooks/              命名码本 .npz（cascade.npz 等）
├── ship_names.json         全局船名映射
└── gallery_meta.json       识别元数据（key → {shipIndex, title}，自动生成）
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
├── data/                        # 整个 gitignored（图片/码本/船名映射，随数据分发）
│   └── groups/group1/           # 原图库 + 测试集（见「数据组织」）
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
