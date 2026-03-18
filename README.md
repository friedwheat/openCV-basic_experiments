# openCV-basic_experiments

OpenCV 基础实验：颜色通道、数据类型与空间域滤波的常见陷阱演示。

## 目录结构

```
.
├── task1/
│   ├── channel/          # 颜色通道与颜色空间陷阱（BGR/RGB）
│   └── Dtype/            # 数据类型陷阱（float32 → uint8）
├── task2/
│   ├── robustness.py     # 鲁棒图像读写
│   ├── test/             # 输入目录
│   └── output/           # 批处理统计与日志
└── task3/
    ├── spatial_filter_edge.py  # 空间域滤波与边缘检测
    └── output_task3/           # 滤波与边缘检测结果
```

## 环境要求

- **OS**：Linux
- **Python**：3.10+（推荐使用 Anaconda）
- **依赖**：

```bash
pip install -r requirements.txt
```

| 包 | 版本 |
|---|---|
| numpy | ≥ 1.23 |
| opencv-python | ≥ 4.7 |
| matplotlib | ≥ 3.7 |

## 注意事项

- `task1` 使用 `cv2.imshow` 与 `matplotlib` 展示图像，需要图形界面环境（SSH / 纯终端不可用）。
