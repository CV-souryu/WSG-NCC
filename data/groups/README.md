# data/groups —— 按组归并的数据集

每组（`groupN`）是一个自洽的识别数据集，含原图库与测试集：

```
data/groups/
├── group1/
│   ├── gallery/       原图库（当前：1/ 2/ 2B/ 三类，共 3362 张）
│   │   ├── 1/         正常卡
│   │   ├── 2/
│   │   └── 2B/        破损卡
│   └── testset/       测试集（cards/ + screens/ + alignment.json + summary.json）
└── group2/            预留：新增一组时按同样结构建立
```

## 新增一组（group2）

```bash
mkdir -p data/groups/group2/gallery data/groups/group2/testset
# 放入原图与测试集后，用新图库重建码本：
#   python -c "from pathlib import Path
#              from cascade_ncc import build_cascade_codebook
#              build_cascade_codebook(sorted(Path('data/groups/group2/gallery').rglob('*.png')), name='cascade-group2')"
```

- `ship_names.json`（`data/` 下）是全局船名映射，各组共用。
- 码本 `cascade.npz` 记录的是**绝对路径**，gallery 移动/换组后必须重建（见上）。
- 测试夹具（`tests/conftest.py`）默认读取 `group1/testset`。
