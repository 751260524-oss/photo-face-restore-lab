import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
THIRD = ROOT / "third_party"


def run(cmd, cwd=None):
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


def restore(gfpgan):
    OUTPUT.mkdir(exist_ok=True)

    for img in INPUT.glob("*"):
        if not img.is_file():
            continue

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