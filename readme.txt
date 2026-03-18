一、工程目标
本工程用于演示三类常见图像处理实践：
1) 颜色通道与颜色空间陷阱（BGR/RGB）
2) 数据类型陷阱（float32 到 uint8）
3) 空间域滤波与边缘检测（含鲁棒读写）

--------------------------------------------------
二、目录结构
exp1/
├─ task1/
│  ├─ channel/
│  │  ├─ channel_trap.py
│  │  └─ print_img                  # 图像 shape/dtype 等信息输出文件
│  └─ Dtype/
│     ├─ data_trap.py
│     └─ outputs/
│        └─ dtype_report.txt        # float32/uint8 对比统计
├─ task2/
│  ├─ robustness.py
│  ├─ test/                          # 输入目录（可注入错误样本）
│  └─ output/
│     ├─ conclusion                  # 批处理统计
│     └─ process_log.csv             # 源文件/角度/输出文件映射
└─ task3/
   ├─ spatial_filter_edge.py
   └─ output_task3/                  # 滤波与边缘检测结果

--------------------------------------------------
三、应用环境
1) 操作系统
- Linux
2) Python
- 建议 Python 3.10+（当前环境可用 Anaconda）

3) 依赖包
- numpy
- opencv-python
- matplotlib（task1 使用）

说明：
- task1 的 cv2.imshow 与 matplotlib 窗口展示通常需要图形界面环境。
- 无图形界面（SSH/纯终端）时，窗口相关功能可能不可用。
