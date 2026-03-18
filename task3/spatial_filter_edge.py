import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

def robust_read_image(path: Path) -> np.ndarray:
    """鲁棒读取图像（支持中文路径）。"""
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在：{path}")
    if not path.is_file():
        raise IsADirectoryError(f"输入路径不是文件：{path}")

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as e:
        raise OSError(f"无法读取文件（权限或路径问题）：{path}") from e

    if data.size == 0:
        raise ValueError(f"文件为空或无法读取：{path}")

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"不是有效图像或图像损坏：{path}")
    return img


def robust_save_image(path: Path, image: np.ndarray) -> None:
    """鲁棒保存图像（支持中文路径）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"编码失败：{path}")
    buf.tofile(str(path))


def add_title(img: np.ndarray, title: str) -> np.ndarray:
    """给图像顶部添加标题，便于对比。"""
    if len(img.shape) == 2:
        canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        canvas = img.copy()

    bar_h = 36
    out = np.zeros((canvas.shape[0] + bar_h, canvas.shape[1], 3), dtype=np.uint8)
    out[bar_h:, :] = canvas
    cv2.putText(
        out,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def make_grid(images: list[np.ndarray], cols: int = 3) -> np.ndarray:
    """生成简单拼图网格。"""
    if not images:
        raise ValueError("images 为空")

    h, w = images[0].shape[:2]
    norm_images = []
    for im in images:
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        norm_images.append(im)

    rows = (len(norm_images) + cols - 1) // cols
    black = np.zeros((h, w, 3), dtype=np.uint8)
    while len(norm_images) < rows * cols:
        norm_images.append(black.copy())

    lines = []
    for r in range(rows):
        line = cv2.hconcat(norm_images[r * cols : (r + 1) * cols])
        lines.append(line)
    return cv2.vconcat(lines)


def run_pipeline(image_path: Path, output_dir: Path) -> None:
    src = robust_read_image(image_path)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 1) 平滑滤波对比
    mean_blur = cv2.blur(src, (5, 5))
    gauss_blur = cv2.GaussianBlur(src, (5, 5), 1.2)
    median_blur = cv2.medianBlur(src, 5)

    robust_save_image(output_dir / "01_original.png", src)
    robust_save_image(output_dir / "02_mean_blur.png", mean_blur)
    robust_save_image(output_dir / "03_gaussian_blur.png", gauss_blur)
    robust_save_image(output_dir / "04_median_blur.png", median_blur)

    # 2) 边缘提取算子
    sobel_x_16s = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y_16s = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    sobel_x_abs = cv2.convertScaleAbs(sobel_x_16s)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y_16s)
    sobel_edge = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)

    lap_16s = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    lap_edge = cv2.convertScaleAbs(lap_16s)

    canny_edge = cv2.Canny(gray, 100, 200)

    robust_save_image(output_dir / "05_gray.png", gray)
    robust_save_image(output_dir / "06_sobel_edge.png", sobel_edge)
    robust_save_image(output_dir / "07_laplacian_edge.png", lap_edge)
    robust_save_image(output_dir / "08_canny_edge.png", canny_edge)

    # 3) 核心陷阱分析（Sobel）
    sobel_a_cv8u = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    sobel_b_cv16s = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)

    # “直接显示 CV_16S”常见错误：负值会被截断/映射异常，这里用截断可视化模拟该问题
    sobel_b_direct_display = np.clip(sobel_b_cv16s, 0, 255).astype(np.uint8)

    sobel_c_abs = cv2.convertScaleAbs(sobel_b_cv16s)

    robust_save_image(output_dir / "09_sobel_a_cv8u.png", sobel_a_cv8u)
    robust_save_image(output_dir / "10_sobel_b_cv16s_direct.png", sobel_b_direct_display)
    robust_save_image(output_dir / "11_sobel_c_cv16s_abs.png", sobel_c_abs)

    # 3) 核心陷阱分析（Laplacian）
    lap_a_cv8u = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    lap_b_cv16s = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    lap_b_direct_display = np.clip(lap_b_cv16s, 0, 255).astype(np.uint8)
    lap_c_abs = cv2.convertScaleAbs(lap_b_cv16s)

    robust_save_image(output_dir / "12_laplacian_a_cv8u.png", lap_a_cv8u)
    robust_save_image(output_dir / "13_laplacian_b_cv16s_direct.png", lap_b_direct_display)
    robust_save_image(output_dir / "14_laplacian_c_cv16s_abs.png", lap_c_abs)

    # 结果总览图
    overview = make_grid(
        [
            add_title(src, "Original"),
            add_title(mean_blur, "Mean Blur"),
            add_title(gauss_blur, "Gaussian Blur"),
            add_title(median_blur, "Median Blur"),
            add_title(sobel_edge, "Sobel Edge"),
            add_title(lap_edge, "Laplacian Edge"),
            add_title(canny_edge, "Canny Edge"),
            add_title(sobel_a_cv8u, "Sobel a: CV_8U"),
            add_title(sobel_b_direct_display, "Sobel b: CV_16S direct"),
            add_title(sobel_c_abs, "Sobel c: CV_16S + abs"),
            add_title(lap_a_cv8u, "Lap a: CV_8U"),
            add_title(lap_b_direct_display, "Lap b: CV_16S direct"),
            add_title(lap_c_abs, "Lap c: CV_16S + abs"),
        ],
        cols=3,
    )
    robust_save_image(output_dir / "00_overview.png", overview)

    print("\n==== 处理完成 ====")
    print(f"输入图像: {image_path}")
    print(f"输出目录: {output_dir}")
   

def main() -> None:
    parser = argparse.ArgumentParser(description="任务三：空间域滤波与边缘提取")
    parser.add_argument(
        "--image",
        type=Path,
        #required=True,
        default=Path("./image.jpg"),
        help="输入图像路径（建议使用细节丰富图像）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output_task3"),
        help="结果输出目录（默认：./output_task3）",
    )
    args = parser.parse_args()

    try:
        run_pipeline(args.image, args.output_dir)
    except (FileNotFoundError, IsADirectoryError, PermissionError, ValueError, OSError) as e:
        print(f"[ERROR] {e}")
        print("[HINT] 请传入可读取的有效图像文件，例如：--image ./test.jpg")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 处理失败：{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
