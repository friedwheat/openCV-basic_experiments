#数据类型陷阱（Float to Uint8）示例
from pathlib import Path

import cv2
import numpy as np


def build_gradient(height: int = 256, width: int = 512) -> np.ndarray:
	#生成二维 float32 渐变图，取值严格在 [0.0, 1.0]
	
	row = np.linspace(0.0, 1.0, width, dtype=np.float32)
	img_f32 = np.tile(row, (height, 1))
	return img_f32


def save_images(out_dir: Path) -> Path:
	out_dir.mkdir(parents=True, exist_ok=True)

	img_f32 = build_gradient()

	# 正确做法：先进行数学映射 [0,1] -> [0,255]，再转 uint8。
	img_u8_correct = np.clip(img_f32 * 255.0, 0.0, 255.0).astype(np.uint8)

	# 错误做法：直接强转 uint8（会截断小数部分）。
	img_u8_wrong = img_f32.astype(np.uint8)

	ok1 = cv2.imwrite(str(out_dir / "gradient_correct_mapping.png"), img_u8_correct)
	ok2 = cv2.imwrite(str(out_dir / "gradient_wrong_direct_cast.png"), img_u8_wrong)

	if not (ok1 and ok2):
		raise RuntimeError("图片保存失败，请检查输出路径或 OpenCV 编码器支持。")

	# 统计信息写入文档，避免终端输出
	unique_wrong = np.unique(img_u8_wrong)
	report_lines = [
		"=== 原始 float32 图像 ===",
		(
			f"dtype={img_f32.dtype}, min={img_f32.min():.6f}, max={img_f32.max():.6f}, "
			f"unique_count(approx)={len(np.unique(img_f32))}"
		),
		"",
		"=== 正确映射后 uint8 ===",
		(
			f"dtype={img_u8_correct.dtype}, min={img_u8_correct.min()}, max={img_u8_correct.max()}, "
			f"unique_count={len(np.unique(img_u8_correct))}"
		),
		"",
		"=== 直接强转后 uint8（错误示例）===",
		(
			f"dtype={img_u8_wrong.dtype}, min={img_u8_wrong.min()}, max={img_u8_wrong.max()}, "
			f"unique_values={unique_wrong.tolist()}"
		),
	]

	report_path = out_dir / "dtype_report.txt"
	report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
	return report_path

	


if __name__ == "__main__":
	# 默认输出到脚本同级目录下的 outputs 文件夹
	save_images(Path(__file__).resolve().parent / "outputs")
	print("图片已保存到 outputs 文件夹，相关统计信息也已写入 dtype_report.txt。")
