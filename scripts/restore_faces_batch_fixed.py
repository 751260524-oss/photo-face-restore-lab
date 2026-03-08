import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
THIRD = ROOT / "third_party"


def run(cmd, cwd=None):
    print("[RUN]", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=cwd, check=True)


def setup_models():
    gfpgan = THIRD / "GFPGAN"

    if not gfpgan.exists():
        run([
            "git", "clone", "--depth", "1",
            "https://github.com/TencentARC/GFPGAN.git",
            str(gfpgan)
        ])

    return gfpgan


def restore(gfpgan: Path):
    OUTPUT.mkdir(exist_ok=True)
    INPUT.mkdir(exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [p for p in INPUT.iterdir() if p.is_file() and p.suffix.lower() in exts]

    if not images:
        print(f"[WARN] No images found in: {INPUT}")
        return

    for img in images:
        out = OUTPUT / img.stem
        out.mkdir(exist_ok=True)

        run([
            sys.executable,
            "inference_gfpgan.py",
            "-i", str(img),
            "-o", str(out),
            "-v", "1.4",
            "-s", "2",
            "--bg_upsampler", "realesrgan"
        ], cwd=gfpgan)


def main():
    THIRD.mkdir(exist_ok=True)
    gfpgan = setup_models()
    restore(gfpgan)


if __name__ == "__main__":
    main()