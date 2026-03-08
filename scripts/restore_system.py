import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
THIRD = ROOT / "third_party"
FINAL = OUTPUT / "final_results"
CMP = OUTPUT / "comparisons"
LOGS = OUTPUT / "logs"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GFPGAN_REPO = "https://github.com/TencentARC/GFPGAN.git"
GFPGAN_VERSION = "1.4"
UPSCALE = "2"
BG_UPSAMPLER = "realesrgan"


def log(*parts: object) -> None:
    print("[INFO]", *parts, flush=True)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_dirs() -> None:
    for p in (INPUT, OUTPUT, THIRD, FINAL, CMP, LOGS):
        p.mkdir(parents=True, exist_ok=True)


def iter_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])


def setup_gfpgan() -> Path:
    gfpgan = THIRD / "GFPGAN"
    if not gfpgan.exists():
        run(["git", "clone", "--depth", "1", GFPGAN_REPO, str(gfpgan)])
    return gfpgan


def make_side_by_side(before_path: Path, after_path: Path, out_path: Path) -> None:
    with Image.open(before_path).convert("RGB") as before_img, Image.open(after_path).convert("RGB") as after_img:
        target_h = max(before_img.height, after_img.height)

        def fit(img: Image.Image) -> Image.Image:
            if img.height == target_h:
                return img
            new_w = max(1, int(img.width * (target_h / img.height)))
            return img.resize((new_w, target_h), Image.LANCZOS)

        before_fit = fit(before_img)
        after_fit = fit(after_img)

        label_h = 40
        gap = 20
        canvas = Image.new(
            "RGB",
            (before_fit.width + after_fit.width + gap, target_h + label_h),
            (245, 245, 245),
        )
        canvas.paste(before_fit, (0, label_h))
        canvas.paste(after_fit, (before_fit.width + gap, label_h))

        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), "Before", fill=(0, 0, 0))
        draw.text((before_fit.width + gap + 10, 10), "After", fill=(0, 0, 0))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, quality=95)


def find_restored_image(image_output_dir: Path, original_stem: str) -> Path | None:
    candidates = [
        image_output_dir / "restored_imgs" / f"{original_stem}.png",
        image_output_dir / "restored_imgs" / f"{original_stem}.jpg",
        image_output_dir / "restored_imgs" / f"{original_stem}.jpeg",
    ]
    for c in candidates:
        if c.exists():
            return c

    restored_dir = image_output_dir / "restored_imgs"
    if restored_dir.exists():
        files = [p for p in restored_dir.iterdir() if p.is_file()]
        if files:
            return sorted(files)[0]
    return None


def restore_one(gfpgan: Path, image_path: Path) -> dict:
    image_output_dir = OUTPUT / image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "inference_gfpgan.py",
        "-i",
        str(image_path),
        "-o",
        str(image_output_dir),
        "-v",
        GFPGAN_VERSION,
        "-s",
        UPSCALE,
        "--bg_upsampler",
        BG_UPSAMPLER,
    ]

    run(cmd, cwd=gfpgan)

    restored = find_restored_image(image_output_dir, image_path.stem)
    if restored is None:
        raise FileNotFoundError(f"No restored image found for {image_path.name}")

    final_ext = restored.suffix.lower() if restored.suffix else ".png"
    final_path = FINAL / f"{image_path.stem}{final_ext}"
    shutil.copy2(restored, final_path)

    cmp_path = CMP / f"{image_path.stem}_before_after.jpg"
    make_side_by_side(image_path, restored, cmp_path)

    return {
        "input": str(image_path.relative_to(ROOT)),
        "restored": str(restored.relative_to(ROOT)),
        "final": str(final_path.relative_to(ROOT)),
        "comparison": str(cmp_path.relative_to(ROOT)),
        "status": "success",
    }


def write_summary(results: list[dict]) -> None:
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Restore Summary",
        "",
        f"- Total: {summary['total']}",
        f"- Success: {summary['success']}",
        f"- Failed: {summary['failed']}",
        "",
    ]
    for item in results:
        lines.append(f"## {Path(item['input']).name}")
        lines.append(f"- Status: {item['status']}")
        if item["status"] == "success":
            lines.append(f"- Final: `{item['final']}`")
            lines.append(f"- Comparison: `{item['comparison']}`")
        else:
            lines.append(f"- Error: `{item['error']}`")
        lines.append("")

    (OUTPUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    images = iter_images(INPUT)
    if not images:
        log(f"No images found in {INPUT}")
        write_summary([])
        return 0

    gfpgan = setup_gfpgan()
    results: list[dict] = []

    for image_path in images:
        log(f"Processing {image_path.name}")
        try:
            results.append(restore_one(gfpgan, image_path))
        except Exception as exc:  # noqa: BLE001
            err_path = LOGS / f"{image_path.stem}.log"
            err_path.write_text(traceback.format_exc(), encoding="utf-8")
            results.append(
                {
                    "input": str(image_path.relative_to(ROOT)),
                    "status": "failed",
                    "error": str(exc),
                    "log": str(err_path.relative_to(ROOT)),
                }
            )
            log(f"Failed: {image_path.name}: {exc}")

    write_summary(results)

    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        log(f"Completed with failures: {len(failed)}")
        return 1

    log("Completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
