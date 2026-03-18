exp1 工程说明（OpenCV + NumPy + Matplotlib）

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

可用如下方式安装（示例）：
- pip:  pip install numpy opencv-python matplotlib
- conda: conda install numpy opencv matplotlib

说明：
- task1 的 cv2.imshow 与 matplotlib 窗口展示通常需要图形界面环境。
- 无图形界面（SSH/纯终端）时，窗口相关功能可能不可用。

--------------------------------------------------
四、各功能使用方法
默认在各脚本所在目录执行命令。

A. task1/channel/channel_trap.py
功能：
- 读取彩色图像
- 输出图像矩阵信息（shape、dtype、文件大小）
- 用 OpenCV 展示 B/G/R 通道
- 用 Matplotlib 演示“直接显示 BGR 导致颜色异常”与“转换 RGB 后正确显示”

运行：
- python channel_trap.py
- python channel_trap.py test.jpg

输出：
- print_img（文本统计）
- GUI 窗口（OpenCV/Matplotlib）


B. task1/Dtype/data_trap.py
功能：
- 构造 [0,1] 的 float32 渐变图
- 对比“正确映射到 [0,255] 后转 uint8”与“直接 astype(uint8) 错误做法”
- 将统计信息写入文件（不依赖终端打印）

运行：
- python data_trap.py

输出（task1/Dtype/outputs/）：
- gradient_correct_mapping.png
- gradient_wrong_direct_cast.png
- dtype_report.txt


C. task2/robustness.py
功能：
- 批量读取输入目录图像，容错跳过异常样本
- 随机旋转（不裁剪）+ resize 到 256x256
- 对错误输入给出 WARN 提示并跳过
- 统计写入 conclusion，明细写入 process_log.csv

运行（默认参数）：
- python robustness.py

常用参数：
- --input-dir  输入目录（默认 ./test）
- --output-dir 输出目录（默认 ./output）
- --inject-error-files 注入错误样本（当前代码默认 True）
- --seed 随机种子（复现实验）

示例：
- python robustness.py --input-dir ./test --output-dir ./output --seed 42

输出（task2/output/）：
- img_0001.png ...（成功处理后的图像）
- conclusion（总数/成功数/跳过数）
- process_log.csv（source_file, angle_deg, output_file）


D. task3/spatial_filter_edge.py
功能：
- 平滑滤波：均值 / 高斯 / 中值
- 边缘提取：Sobel / Laplacian / Canny
- 演示梯度算子数据类型陷阱（CV_8U vs CV_16S + abs）
- 输出总览拼图与分步结果

运行（默认 image.jpg）：
- python spatial_filter_edge.py

指定输入：
- python spatial_filter_edge.py --image ./image.jpg --output-dir ./output_task3

输出（task3/output_task3/）：
- 00_overview.png
- 01_original.png ... 14_laplacian_c_cv16s_abs.png

--------------------------------------------------
五、常见问题
1) "无法解析导入 cv2 / matplotlib"
- 多为解释器环境未切换到安装依赖的 Python。
- 请在当前运行环境中安装依赖，或在编辑器中切换解释器。

2) task1 无法弹窗
- 当前会话无 GUI（远程/无桌面）时，imshow/show 可能不可用。
- 建议在桌面会话下运行，或改为保存图片查看。

3) 输入目录全是错误样本
- robustness.py 会输出 WARN 并跳过，对应统计写入 conclusion。

--------------------------------------------------
六、建议执行顺序
1) 先跑 task1：理解 BGR/RGB 与显示差异
2) 再跑 task1 Dtype：理解 float32->uint8 的正确映射
3) 跑 task2：验证鲁棒批处理与错误输入跳过机制
4) 最后跑 task3：完成滤波、边缘、数据类型陷阱综合实验
