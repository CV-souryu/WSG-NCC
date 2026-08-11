> ⚠️ **OUTDATED / 历史记录**：本文档描述的是已删除的 spiral / concentric 路径
> （`VerticalShipCardRecognizer`、`concentric_codebook.py` 等已不存在），
> 与当前 cascade-only 实现无关。当前 API 见 `README.md`。

# BlueCard / M1 船只卡识别实验记录

## 目标

用 M1 图库识别 `data/queries/blue_cards/` 下的 6 张 BlueCard
（card0–card5，1x/4x 共 12 张原图，其中 6 张独立卡面）。
已知 card0 应识别为 **M_NORMAL_214**。

## 数据与码本

- 图库：`data/gallery/M1/`，1596 张 `M_NORMAL_*.png`，218×585 RGBA
- 实验码本：M1c60（M1 裁掉下方 2/5 后临时重建，位于 `/tmp/M1_crop60`）
- BlueCard：`data/queries/blue_cards/`，138×347 RGB

## 最终推荐配置（同心圆码点）

- 圆心：正中心（124×240 画布上的 61.5, 119.5）
- 点阵：同心圆，半径等比增长：`r0 = 8`，`q = 1.20`
- 圈数：自动生成，直到圆半径超过图片最远角（约 134.4px），实际 16 圈
- 每圈点数：线性递减 **44 → 6**（内圈密、外圈疏）
- 像素合成：**2×2 邻域平均**
- 码点总数：**401**
- 公共码点（透明度有效 ≥90%）：**311**

## 6 张卡最终结果

| 图片 | Top1 | Top2 | Top3 | Top1 NCC | 对比度 |
|---|---|---|---|---:|---:|
| card0 | 214 | 343 | 52 | 0.9484 | 0.5003 |
| card1 | 206 | 354 | 1086 | 0.9826 | 0.4390 |
| card2 | 108 | 223 | 461_4 | 0.9095 | 0.5328 |
| card3 | 29 | 1055 | 226 | 0.9567 | 0.4787 |
| card4 | 5 | 1219 | 162 | 0.9590 | 0.4642 |
| card5 | 42 | 1219 | 1483 | 0.9757 | 0.4906 |

6/6 全部 Top1 命中；平均对比度约 0.484，最低 0.439。

## 关键实验结论

1. **不切蓝边**：BlueCard 蓝色边框只有 1–2px，自动裁蓝边反而让 214 掉出 Top5。
2. **源图预处理**：宽度缩到 **194**（218−12−12），顶部留 6px 透明边，
   画布 218×351（M1c60 高度），下方超出部分直接裁断。
3. **点阵形状**：同心圆（内密外疏）> 同心矩形 > 螺旋 > 矩形网格。
   不规则/非均匀取点对偏移更鲁棒，公共码点更聚焦。
4. **像素合成**：2×2 比 1×1 更抗偏移：
   - ±2px 偏移：1×1 最低对比度 0.037，2×2 为 0.146，3×3 为 0.200
   - ±0.5/1px 偏移：2×2 平均对比度最高（0.408）
5. **取点策略**：线性 44→6 优于固定点数和等比递减。
6. **参数网格（2×2）**：r0=8、q=1.20、线性 44→6 综合最优；
   r0=6、q=1.25（351 点）平均对比度最高但最差卡略低。

## 生成文件

- 叠图/码点图：`data/queries/blue_cards_compare/`
- 网格测试 CSV：
  - `data/queries/m1c60_grid_test.csv`（coverage × spacing × pixel）
  - `data/queries/m1c60_rings2x2_grid.csv`（r0 × q × 取点策略，2×2）

## 状态

同心圆码本已正式实现：

- 模块：`spiral_ncc/concentric_codebook.py`
- 构建脚本：`scripts/build_concentric_codebook.py`
- 默认参数：`r0=8, q=1.20, 线性 44→6, 2×2, 圆心 (0.5,0.5)`
- 正式码本：`artifacts/codebooks/concentric.npz`（M1c60，401 点 / 311 公共）
- 使用：`VerticalShipCardRecognizer("concentric")` 可直接加载
