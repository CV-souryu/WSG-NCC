# 码本目录

本目录随仓库分发默认码本 `cascade.npz`；`assets/gallery_meta.json` 也在仓库中。
`data/` 下其余内容（图片、测试集、船名映射等）仍由 Git 忽略，按需单独分发。

## cascade.npz

- 全码本：`top_fraction=1.0`
- 构建参数：`step=2`、`ncc_step=8`、`ncc_pool=9`、
  H16S2L2 × 3×3 直方图、`trim_blue=True`、`shift_y=4`、
  `fit_width=True`
- 基于 3359 张 `124×240` gallery 图构建
- 不保存 `normed8`，CPU 首次识别时由 `get_normed8()` 惰性生成
- 大小约 6.1 MB

用法：

```python
from cascade_ncc import CascadeShipRecognizer, load_cascade_codebook

rec = CascadeShipRecognizer("cascade")          # 默认从本目录加载
cb = load_cascade_codebook("cascade")
```

本地其它码本（`cascade-full`、`cascade-top60`、`cascade-top40-u33`、
`cascade-top50`）不随仓库分发。
