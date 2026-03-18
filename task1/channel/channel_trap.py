import cv2
import matplotlib
import numpy as np
import os
from pathlib import Path


def task1_channel_trap(image_path)->bool:
    
    #演示 OpenCV BGR 和 Matplotlib RGB 颜色空间的陷阱
    
    try:
        # 1. 使用 cv2.imread() 读取彩色图像
        if not os.path.isfile(image_path):
            print(f"错误：文件 '{image_path}' 不存在")
            return False
        
        img_bgr = cv2.imread(image_path)
        
        if img_bgr is None:
            print(f"错误：无法读取图像文件 {image_path}")
            return False

        # 2. 打印并解释其 shape 与 dtype
        # shape: (高度, 宽度, 通道数), dtype: 像素数据类型（通常为 uint8）
        img_info_lines = [
            "=" * 60,
            "【图像矩阵信息】",
            f"- Shape (H, W, C): {img_bgr.shape}",
            f"- Data Type: {img_bgr.dtype}",
            f"- 文件大小: {os.path.getsize(image_path) / 1024:.2f} KB",
            "=" * 60,
        ]
        for line in img_info_lines:
            print(line)

        # 同步写入文件：print_img
        print_img_path = Path(__file__).resolve().parent / "print_img"
        try:
            with open(print_img_path, "w", encoding="utf-8") as f:
                f.write("\n".join(img_info_lines) + "\n")
        except Exception as e:
            print(f"[WARN] 写入 {print_img_path} 失败: {e}")

        # 3. 使用 cv2.imshow() 显示图像及各颜色通道
        print("\n【OpenCV 显示阶段】")
        print("- 显示原始 BGR 图像和各颜色通道...")
        
        cv2.imshow("OpenCV BGR (Correct in CV Window)", img_bgr)
        
        b, g, r = cv2.split(img_bgr)
        cv2.imshow("Blue Channel", b)
        cv2.imshow("Green Channel", g)
        cv2.imshow("Red Channel", r)
        
        print("[提示] 请查看弹出窗口，按任意键关闭窗口以继续运行 Matplotlib 部分...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 关键修复：OpenCV(Qt)会设置 Qt 插件路径，可能导致 Matplotlib 的 Qt 后端崩溃
        # 切换 Matplotlib 到 Tk 后端并延迟导入 pyplot，避免与 cv2 的 Qt 插件冲突
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        os.environ.pop("QT_QPA_FONTDIR", None)
        try:
            matplotlib.use("TkAgg", force=True)
        except Exception:
            matplotlib.use("Agg", force=True)
            print("[提示] TkAgg 后端不可用，已切换为 Agg（将不弹出 Matplotlib 窗口）。")
        
        #延迟导入 pyplot，确保后端设置生效
        import matplotlib.pyplot as plt

        # 4. 触发陷阱：直接将 BGR 数据传给 matplotlib.pyplot.imshow()
        print("\n【Matplotlib 颜色空间对比】")

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.title("Abnormal: BGR in Matplotlib ")
        plt.imshow(img_bgr)  # 此时红色变蓝色，肤色会发青
        plt.axis('off')

        # 5. 给出正确修复代码并展示
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        plt.subplot(1, 2, 2)
        plt.title("Fixed: Converted to RGB")
        plt.imshow(img_rgb)
        plt.axis('off')

        print("[提示] 正在显示 Matplotlib 对比图...")
        plt.tight_layout()
        plt.show()
        
        
        return True
        
    except Exception as e:
        print(f"\n 发生异常: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"堆栈信息:\n{traceback.format_exc()}")
        return False
    


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = 'test.jpg'
    
    print(f"开始处理图像: {test_image}")
    success = task1_channel_trap(test_image)
    
    if success:
        print("\n 处理完成")
    else:
        print("\n 处理失败")