import argparse
import random
from pathlib import Path

import numpy as np
import cv2

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def create_error_files(test_dir: Path) -> None:
	#在已有数据目录中注入错误样本文件。
	
	test_dir.mkdir(parents=True, exist_ok=True)

	# 无关文本文件
	(test_dir / "notes.txt").write_text("this is not an image\n", encoding="utf-8")
	(test_dir / "report.md").write_text("# fake document\n", encoding="utf-8")

	# 伪造后缀文件（扩展名看似图片，但内容不是图片）
	(test_dir / "fake_image.jpg").write_text("not a real jpg", encoding="utf-8")

	# 损坏图片文件（只写入部分PNG头）
	with open(test_dir / "corrupted.png", "wb") as f:
		f.write(b"\x89PNG\r\n\x1a\n\x00\x00")

	# 空文件（扩展名是图片）
	(test_dir / "empty.jpeg").write_bytes(b"")

	# 一个子目录（应被跳过）
	(test_dir / "subfolder").mkdir(exist_ok=True)

	print(f"[INFO] 错误样本已注入：{test_dir}")


def robust_read_image(file_path: Path) -> np.ndarray | None:
	#鲁棒读取图像，失败返回None
	try:
		data = np.fromfile(str(file_path), dtype=np.uint8)
		if data.size == 0:
			print(f"[WARN] 跳过（空文件）：{file_path.name}")
			return None
		img = cv2.imdecode(data, cv2.IMREAD_COLOR)
		if img is None:
			print(f"[WARN] 跳过（非图像或损坏）：{file_path.name}")
			return None
		return img
	except Exception as e:
		print(f"[WARN] 跳过（读取异常）：{file_path.name}, 原因：{e}")
		return None


def rotate_image_no_crop(image: np.ndarray, angle_deg: float) -> np.ndarray:
	#旋转图像且不裁剪内容，空缺区域黑色填充。
	h, w = image.shape[:2]
	center = (w / 2.0, h / 2.0)

	# 先按原尺寸计算旋转矩阵
	m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
	cos = abs(m[0, 0])
	sin = abs(m[0, 1])

	# 计算完整包围盒尺寸，确保零裁剪
	new_w = int(np.ceil((h * sin) + (w * cos)))
	new_h = int(np.ceil((h * cos) + (w * sin)))

	# 平移矩阵，使旋转后的图像中心落在新图像中心
	m[0, 2] += (new_w / 2.0) - center[0]
	m[1, 2] += (new_h / 2.0) - center[1]

	rotated = cv2.warpAffine(
		image,
		m,
		(new_w, new_h),
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_CONSTANT,
		borderValue=(0, 0, 0),
	)
	return rotated


def process_images(input_dir: Path, output_dir: Path, seed: int | None = None) -> None:
	"""遍历目录进行鲁棒批处理。"""
	if seed is not None:
		random.seed(seed)

	output_dir.mkdir(parents=True, exist_ok=True)

	success_count = 0
	skip_count = 0
	total_count = 0
	records: list[tuple[str, float, str]] = []

	for item in sorted(input_dir.iterdir(), key=lambda p: p.name.lower()):
		total_count += 1

		if not item.is_file():
			skip_count += 1
			print(f"[WARN] 跳过（非文件）：{item.name}")
			continue

		# 先尝试读取，不依赖扩展名，增强鲁棒性
		img = robust_read_image(item)
		if img is None:
			skip_count += 1
			continue

		try:
			angle = random.uniform(0.0, 360.0)
			rotated = rotate_image_no_crop(img, angle)
			resized = cv2.resize(rotated, (256, 256), interpolation=cv2.INTER_AREA)

			out_name = f"img_{success_count + 1:04d}.png"
			out_path = output_dir / out_name
			ok = cv2.imwrite(str(out_path), resized)
			if not ok:
				raise RuntimeError("cv2.imwrite 返回 False")

			success_count += 1
			records.append((item.name, angle, out_name))
			#print(f"[OK] {item.name} -> {out_name}, angle={angle:.2f}")
		except Exception as e:
			skip_count += 1
			print(f"[WARN] 处理失败并跳过：{item.name}, 原因：{e}")

	
	# 统计信息写入 conclusion 文件
	conclusion_file = output_dir / "conclusion"
	conclusion_lines = [
		"========== 处理统计 ==========",
		f"输入目录：{input_dir}",
		f"输出目录：{output_dir}",
		f"总条目数：{total_count}",
		f"成功数：{success_count}",
		f"跳过数：{skip_count}",
		"==============================",
	]
	conclusion_file.write_text("\n".join(conclusion_lines) + "\n", encoding="utf-8")
	print(f"[INFO] 统计信息已保存：{conclusion_file}")

	# 追踪记录保存到CSV
	record_file = output_dir / "process_log.csv"
	with open(record_file, "w", encoding="utf-8") as f:
		f.write("source_file,angle_deg,output_file\n")
		for src, angle, dst in records:
			f.write(f"{src},{angle:.6f},{dst}\n")
	print(f"[INFO] 处理日志已保存：{record_file}")


def main() -> None:
	parser = argparse.ArgumentParser(description="任务二：鲁棒的图像批处理流水线")
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=Path("./test"),
		help="输入目录（默认：./test）",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("./output"),
		help="输出目录（默认：./output）",
	)
	parser.add_argument(
		"--inject-error-files",
		action="store_true",
		default=True,
		help="在已有输入目录中仅注入错误样本（不新建测试图片）",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="随机种子（可选，便于复现实验）",
	)
	args = parser.parse_args()

	if args.inject_error_files:
		create_error_files(args.input_dir)

	if not args.input_dir.exists() or not args.input_dir.is_dir():
		raise FileNotFoundError(f"输入目录不存在或不是目录：{args.input_dir}")

	process_images(args.input_dir, args.output_dir, seed=args.seed)


if __name__ == "__main__":
	main()
