import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def run(cmd: List[str], cwd: str | None = None) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_images(input_dir: Path) -> List[Path]:
    files = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return files


def clone_repo_if_needed(repo_dir: Path) -> None:
    if repo_dir.exists():
        print(f"[INFO] Repo already exists: {repo_dir}")
        return

    run([
        "git",
        "clone",
        "--depth", "1",
        "https://github.com/TencentARC/GFPGAN.git",
        str(repo_dir)
    ])


def install_dependencies(repo_dir: Path) -> None:
    # 安装 GFPGAN 及其依赖
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=str(repo_dir))
    run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=str(repo_dir))

    # 有些环境下额外装一下 basicsr/realesrgan 更稳
    run([sys.executable, "-m", "pip", "install", "basicsr", "facexlib", "realesrgan"])


def ensure_weights(repo_dir: Path) -> None:
    weights_dir = repo_dir / "gfpgan" / "weights"
    ensure_dir(weights_dir)

    # 直接使用 GFPGAN 自带下载脚本最稳
    inference_script = repo_dir / "inference_gfpgan.py"
    if not inference_script.exists():
        raise FileNotFoundError(f"Missing script: {inference_script}")


def process_one_image(repo_dir: Path, input_image: Path, output_dir: Path) -> None:
    stem_dir = output_dir / input_image.stem
    ensure_dir(stem_dir)

    # GFPGAN 输出目录
    temp_out = stem_dir / "_gfpgan_raw"
    ensure_dir(temp_out)

    # 用 GFPGAN 的整图处理模式：
    # - version 1.4 比较稳
    # - upscale 2 先别太大，避免过度假脸
    # - bg_upsampler realesrgan，背景只做放大，不做重绘
    cmd = [
        sys.executable,
        "inference_gfpgan.py",
        "-i", str(input_image.resolve()),
        "-o", str(temp_out.resolve()),
        "-v", "1.4",
        "-s", "2",
        "--bg_upsampler", "realesrgan",
    ]
    run(cmd, cwd=str(repo_dir))

    # GFPGAN 默认输出路径一般是：
    # results/restored_imgs/xxx.png
    restored_dir = temp_out / "restored_imgs"
    if not restored_dir.exists():
        raise FileNotFoundError(f"Expected output directory not found: {restored_dir}")

    restored_files = sorted([p for p in restored_dir.iterdir() if p.is_file()])
    if not restored_files:
        raise FileNotFoundError(f"No restored file found in: {restored_dir}")

    final_target = stem_dir / f"{input_image.stem}_restored{restored_files[0].suffix.lower()}"
    shutil.copy2(restored_files[0], final_target)

    print(f"[DONE] {input_image.name} -> {final_target}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    input_dir = project_root / "input"
    output_dir = project_root / "output"
    third_party_dir = project_root / "third_party"
    repo_dir = third_party_dir / "GFPGAN"

    ensure_dir(output_dir)
    ensure_dir(third_party_dir)

    images = find_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No input images found in: {input_dir}")

    print(f"[INFO] Found {len(images)} image(s).")

    clone_repo_if_needed(repo_dir)
    install_dependencies(repo_dir)
    ensure_weights(repo_dir)

    for img in images:
        process_one_image(repo_dir, img, output_dir)

    print(f"[DONE] Processed {len(images)} image(s). Results saved to: {output_dir}")


if __name__ == "__main__":
    main()