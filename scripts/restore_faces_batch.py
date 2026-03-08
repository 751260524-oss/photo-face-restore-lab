#!/usr/bin/env python3
"""
Professional conservative face-only restoration batch tool.

- Input: all JPG/JPEG files under input/
- Output per image: output/<image_stem>/v01.jpg ... v24.jpg, contact_sheet.jpg, params.csv
- Restoration target: conservative real-detail enhancement and natural blending only
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


Box = Tuple[int, int, int, int]
DEBUG_FACE_SAMPLES = 12
MAD_TOO_WEAK_THRESHOLD = 1.5
MAD_OK_THRESHOLD = 4.5


@dataclass(frozen=True)
class VariantParams:
    group: str
    denoise: float
    local_contrast: float
    sharpen: float
    blend_strength: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative face-only restoration batch tool")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--face-pad", type=float, default=0.22)
    parser.add_argument("--sheet-thumb-width", type=int, default=340)
    return parser.parse_args()


def iter_jpg_images(input_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}:
            paths.append(p)
    return paths


def clip_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return (x1, y1, x2, y2)


def expand_box(box: Box, pad_ratio: float, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    return clip_box((x1 - px, y1 - py, x2 + px, y2 + py), width, height)


def _inner_face_box(box: Box, width: int, height: int, ratio: float = 0.82) -> Box:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bw = max(2, int((x2 - x1) * ratio))
    bh = max(2, int((y2 - y1) * ratio))
    return clip_box((cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2), width, height)


def box_area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / float(max(1, union))


def merge_boxes(boxes: Sequence[Tuple[Box, int]], iou_threshold: float = 0.35) -> List[Box]:
    if not boxes:
        return []

    sorted_items = sorted(boxes, key=lambda item: (item[1], box_area(item[0])), reverse=True)
    kept: List[Box] = []

    for candidate, _score in sorted_items:
        duplicate = False
        for chosen in kept:
            if iou(candidate, chosen) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    kept.sort(key=box_area, reverse=True)
    return kept


def _load_cascade(name: str) -> cv2.CascadeClassifier:
    path = Path(cv2.data.haarcascades) / name
    detector = cv2.CascadeClassifier(str(path))
    if detector.empty():
        raise RuntimeError(f"Failed to load cascade: {path}")
    return detector


def _is_likely_face_box(box: Box, gray: np.ndarray) -> bool:
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    area_ratio = (bw * bh) / float(w * h)

    if bw < 24 or bh < 24:
        return False
    if area_ratio < 0.0006 or area_ratio > 0.28:
        return False

    cx = (x1 + x2) * 0.5 / w
    cy = (y1 + y2) * 0.5 / h
    if cy < 0.03 or cy > 0.97:
        return False
    if cx < 0.01 or cx > 0.99:
        return False

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    std = float(np.std(roi))
    if std < 10.0:
        return False

    return True


def detect_faces(image_bgr: np.ndarray, pad_ratio: float = 0.22) -> List[Box]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    detector_defs = [
        ("haarcascade_frontalface_default.xml", 7),
        ("haarcascade_frontalface_alt2.xml", 9),
        ("haarcascade_profileface.xml", 5),
    ]
    detectors: List[Tuple[cv2.CascadeClassifier, int]] = [(_load_cascade(name), score) for name, score in detector_defs]

    candidates: List[Tuple[Box, int]] = []
    for detector, base_score in detectors:
        faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(24, 24), flags=cv2.CASCADE_SCALE_IMAGE)
        for x, y, fw, fh in faces:
            box = clip_box((int(x), int(y), int(x + fw), int(y + fh)), w, h)
            if _is_likely_face_box(box, gray):
                candidates.append((expand_box(box, pad_ratio, w, h), base_score))

    gray_2x = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    for detector, base_score in detectors:
        faces = detector.detectMultiScale(gray_2x, scaleFactor=1.08, minNeighbors=5, minSize=(32, 32), flags=cv2.CASCADE_SCALE_IMAGE)
        for x, y, fw, fh in faces:
            sx1 = int(round(x / 2.0))
            sy1 = int(round(y / 2.0))
            sx2 = int(round((x + fw) / 2.0))
            sy2 = int(round((y + fh) / 2.0))
            box = clip_box((sx1, sy1, sx2, sy2), w, h)
            if _is_likely_face_box(box, gray):
                candidates.append((expand_box(box, pad_ratio, w, h), base_score + 1))

    return merge_boxes(candidates, iou_threshold=0.33)


def restore_face_patch(face_patch: np.ndarray, params: VariantParams) -> np.ndarray:
    h, w = face_patch.shape[:2]
    upscale = 3 if min(h, w) < 90 else 2

    work = cv2.resize(face_patch, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    h_denoise = max(0.9, min(2.8, params.denoise))
    work = cv2.fastNlMeansDenoisingColored(work, None, h=h_denoise, hColor=h_denoise, templateWindowSize=7, searchWindowSize=21)

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_clip = max(1.03, min(1.78, 1.0 + params.local_contrast))
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    mix = np.clip(params.local_contrast, 0.04, 0.36)
    l = cv2.addWeighted(l, 1.0 - mix, l2, mix, 0)
    work = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(work, (0, 0), sigmaX=0.85, sigmaY=0.85)
    amount = np.clip(params.sharpen, 0.06, 0.38)
    work = cv2.addWeighted(work, 1.0 + amount, blur, -amount, 0)
    work = np.clip(work, 0, 255).astype(np.uint8)

    restored = cv2.resize(work, (w, h), interpolation=cv2.INTER_AREA)
    return restored


def match_patch_tone(base_roi: np.ndarray, patch_roi: np.ndarray) -> np.ndarray:
    base_lab = cv2.cvtColor(base_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    patch_lab = cv2.cvtColor(patch_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    out = patch_lab.copy()
    for c in range(3):
        b = base_lab[:, :, c]
        p = patch_lab[:, :, c]
        b_mean, b_std = float(np.mean(b)), float(np.std(b))
        p_mean, p_std = float(np.mean(p)), float(np.std(p))
        p_std = max(1.0, p_std)

        matched = (p - p_mean) * (b_std / p_std) + b_mean
        if c == 0:
            alpha = 0.22
            matched = p * (1.0 - alpha) + matched * alpha
        else:
            alpha = 0.16
            matched = p * (1.0 - alpha) + matched * alpha
        out[:, :, c] = np.clip(matched, 0, 255)

    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _ellipse_mask(height: int, width: int, feather: float = 0.12) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (max(1, int(width * 0.44)), max(1, int(height * 0.48)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    k = int(max(7, ((min(height, width) * feather) // 2) * 2 + 1))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def blend_patch(base_img: np.ndarray, patch: np.ndarray, box: Box, blend_strength: float) -> None:
    x1, y1, x2, y2 = box
    roi = base_img[y1:y2, x1:x2]
    if roi.size == 0 or patch.shape[:2] != roi.shape[:2]:
        return

    h, w = roi.shape[:2]
    mask_u8 = _ellipse_mask(h, w)
    mask = (mask_u8.astype(np.float32) / 255.0)[:, :, None]

    alpha = np.clip(blend_strength, 0.46, 0.78)
    blended = roi.astype(np.float32) * (1.0 - mask * alpha) + patch.astype(np.float32) * (mask * alpha)
    blended_u8 = np.clip(blended, 0, 255).astype(np.uint8)

    center = (x1 + w // 2, y1 + h // 2)
    clone = cv2.seamlessClone(blended_u8, base_img, mask_u8, center, cv2.NORMAL_CLONE)
    clone_roi = clone[y1:y2, x1:x2]

    final_mix = 0.18
    base_img[y1:y2, x1:x2] = cv2.addWeighted(blended_u8, 1.0 - final_mix, clone_roi, final_mix, 0)


def _label_pil(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, font: ImageFont.ImageFont) -> None:
    draw.rectangle([(x, y), (x + 110, y + 28)], fill=(0, 0, 0))
    draw.text((x + 8, y + 6), text, fill=(255, 255, 255), font=font)


def _extract_face_zoom_samples(original_bgr: np.ndarray, face_boxes: Sequence[Box]) -> List[Tuple[str, np.ndarray]]:
    h, w = original_bgr.shape[:2]

    def crop_zoom(box: Box, zoom: float = 2.0) -> np.ndarray:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bw = max(20, int((x2 - x1) / zoom))
        bh = max(20, int((y2 - y1) / zoom))
        z = clip_box((cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2), w, h)
        zx1, zy1, zx2, zy2 = z
        region = original_bgr[zy1:zy2, zx1:zx2]
        if region.size == 0:
            return original_bgr
        return cv2.resize(region, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)

    if not face_boxes:
        fallback = [
            ("center_2x", clip_box((w // 2 - w // 8, h // 2 - h // 8, w // 2 + w // 8, h // 2 + h // 8), w, h)),
            ("left_2x", clip_box((w // 6 - w // 10, h // 2 - h // 8, w // 6 + w // 10, h // 2 + h // 8), w, h)),
            ("right_2x", clip_box((5 * w // 6 - w // 10, h // 2 - h // 8, 5 * w // 6 + w // 10, h // 2 + h // 8), w, h)),
            ("random_2x", clip_box((w // 3 - w // 10, h // 3 - h // 10, w // 3 + w // 10, h // 3 + h // 10), w, h)),
        ]
        out: List[Tuple[str, np.ndarray]] = []
        for name, b in fallback:
            out.append((name, cv2.resize(original_bgr[b[1]:b[3], b[0]:b[2]], (260, 260), interpolation=cv2.INTER_CUBIC)))
        return out

    centers = [((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5) for b in face_boxes]
    c_idx = int(np.argmin([(cx - w * 0.5) ** 2 + (cy - h * 0.5) ** 2 for cx, cy in centers]))
    l_idx = int(np.argmin([cx for cx, _ in centers]))
    r_idx = int(np.argmax([cx for cx, _ in centers]))
    rng = np.random.default_rng(42)
    rand_idx = int(rng.integers(0, len(face_boxes)))

    picks = [
        ("center_2x", face_boxes[c_idx]),
        ("left_2x", face_boxes[l_idx]),
        ("right_2x", face_boxes[r_idx]),
        ("random_2x", face_boxes[rand_idx]),
    ]
    return [(name, crop_zoom(box, zoom=2.0)) for name, box in picks]


def build_contact_sheet(
    original_bgr: np.ndarray,
    versions: Sequence[np.ndarray],
    out_path: Path,
    thumb_width: int = 340,
    face_boxes: Sequence[Box] | None = None,
) -> None:
    pil_font = ImageFont.load_default()

    all_items = [("original", original_bgr)] + [(f"v{i:02d}", img) for i, img in enumerate(versions, start=1)]

    thumbs: List[Tuple[str, Image.Image]] = []
    max_h = 0
    for label, bgr in all_items:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        ratio = thumb_width / float(img.width)
        th = int(img.height * ratio)
        thumb = img.resize((thumb_width, th), Image.Resampling.LANCZOS)
        thumbs.append((label, thumb))
        max_h = max(max_h, th)

    zoom_items = _extract_face_zoom_samples(original_bgr, face_boxes or [])
    zoom_thumb_size = max(180, thumb_width // 2)
    zoom_thumbs: List[Tuple[str, Image.Image]] = []
    for label, bgr in zoom_items:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        zimg = Image.fromarray(rgb).resize((zoom_thumb_size, zoom_thumb_size), Image.Resampling.LANCZOS)
        zoom_thumbs.append((label, zimg))

    label_h = 34
    tile_h = max_h + label_h
    columns = 6
    version_rows = 4
    margin = 20
    gap = 14

    zoom_row_h = zoom_thumb_size + label_h
    sheet_w = margin * 2 + columns * thumb_width + (columns - 1) * gap
    sheet_h = margin * 4 + tile_h * (1 + version_rows) + gap * (1 + version_rows) + zoom_row_h
    sheet = Image.new("RGB", (sheet_w, sheet_h), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)

    orig_label, orig_thumb = thumbs[0]
    top_y = margin
    ox = (sheet_w - orig_thumb.width) // 2
    oy = top_y
    sheet.paste(orig_thumb, (ox, oy))
    _label_pil(draw, orig_label, ox, oy + max_h, pil_font)

    zy = margin * 2 + tile_h
    total_zw = len(zoom_thumbs) * zoom_thumb_size + (len(zoom_thumbs) - 1) * gap
    zx0 = (sheet_w - total_zw) // 2
    for i, (label, thumb) in enumerate(zoom_thumbs):
        x = zx0 + i * (zoom_thumb_size + gap)
        sheet.paste(thumb, (x, zy))
        _label_pil(draw, label, x, zy + zoom_thumb_size, pil_font)

    start_y = margin * 3 + tile_h + zoom_row_h + gap
    version_thumbs = thumbs[1:]
    idx = 0
    for r in range(version_rows):
        y = start_y + r * (tile_h + gap)
        for c in range(columns):
            if idx >= len(version_thumbs):
                break
            label, thumb = version_thumbs[idx]
            x = margin + c * (thumb_width + gap)
            sheet.paste(thumb, (x, y))
            _label_pil(draw, label, x, y + max_h, pil_font)
            idx += 1

    sheet.save(out_path, quality=94)


def build_variants() -> List[VariantParams]:
    groups: Dict[str, List[VariantParams]] = {
        "A": [
            VariantParams("A", 1.1, 0.12, 0.10, 0.52),
            VariantParams("A", 1.2, 0.14, 0.12, 0.54),
            VariantParams("A", 1.3, 0.16, 0.14, 0.56),
            VariantParams("A", 1.4, 0.18, 0.16, 0.58),
            VariantParams("A", 1.5, 0.20, 0.18, 0.60),
            VariantParams("A", 1.6, 0.22, 0.20, 0.62),
        ],
        "B": [
            VariantParams("B", 1.2, 0.20, 0.18, 0.60),
            VariantParams("B", 1.3, 0.22, 0.20, 0.62),
            VariantParams("B", 1.4, 0.24, 0.22, 0.64),
            VariantParams("B", 1.5, 0.26, 0.24, 0.66),
            VariantParams("B", 1.6, 0.28, 0.26, 0.68),
            VariantParams("B", 1.7, 0.30, 0.28, 0.70),
        ],
        "C": [
            VariantParams("C", 1.3, 0.28, 0.24, 0.68),
            VariantParams("C", 1.4, 0.30, 0.26, 0.70),
            VariantParams("C", 1.5, 0.32, 0.28, 0.72),
            VariantParams("C", 1.6, 0.34, 0.30, 0.74),
            VariantParams("C", 1.7, 0.36, 0.32, 0.76),
            VariantParams("C", 1.8, 0.38, 0.34, 0.78),
        ],
        "D": [
            VariantParams("D", 1.4, 0.32, 0.28, 0.74),
            VariantParams("D", 1.5, 0.34, 0.30, 0.76),
            VariantParams("D", 1.6, 0.36, 0.32, 0.78),
            VariantParams("D", 1.7, 0.38, 0.34, 0.80),
            VariantParams("D", 1.8, 0.40, 0.36, 0.82),
            VariantParams("D", 1.9, 0.42, 0.38, 0.84),
        ],
    }

    ordered: List[VariantParams] = []
    for g in ("A", "B", "C", "D"):
        ordered.extend(groups[g])
    return ordered


def _make_before_after(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    h = max(before.shape[0], after.shape[0])
    w = max(before.shape[1], after.shape[1])
    b = cv2.copyMakeBorder(before, 0, h - before.shape[0], 0, w - before.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    a = cv2.copyMakeBorder(after, 0, h - after.shape[0], 0, w - after.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    pair = cv2.hconcat([b, a])
    cv2.putText(pair, "before | after", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return pair


def _build_diff_map(original: np.ndarray, restored: np.ndarray) -> Tuple[np.ndarray, float]:
    diff = cv2.absdiff(restored, original)
    mad = float(np.mean(diff))
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(restored, 0.45, heat, 0.55, 0)
    return overlay, mad


def _process_single_variant(
    original: np.ndarray,
    faces: Sequence[Box],
    params: VariantParams,
) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray, np.ndarray]]]:
    out = original.copy()
    debug_pairs: List[Tuple[int, np.ndarray, np.ndarray]] = []
    h, w = out.shape[:2]

    for face_idx, box in enumerate(faces, start=1):
        ix1, iy1, ix2, iy2 = _inner_face_box(box, w, h, ratio=0.82)
        base_roi = out[iy1:iy2, ix1:ix2]
        if base_roi.size == 0:
            continue

        before = base_roi.copy()
        patch = restore_face_patch(base_roi, params)
        patch = match_patch_tone(base_roi, patch)
        blend_patch(out, patch, (ix1, iy1, ix2, iy2), params.blend_strength)
        after = out[iy1:iy2, ix1:ix2].copy()
        debug_pairs.append((face_idx, before, after))

    return out, debug_pairs


def _write_params_csv(csv_path: Path, variants: Sequence[VariantParams], detector_count: int) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "denoise", "local_contrast", "sharpen", "blend_strength", "detector_count"])
        for i, p in enumerate(variants, start=1):
            writer.writerow([f"v{i:02d}", f"{p.denoise:.3f}", f"{p.local_contrast:.3f}", f"{p.sharpen:.3f}", f"{p.blend_strength:.3f}", detector_count])


def process_image(img_path: Path, out_root: Path, variants: Sequence[VariantParams], face_pad: float, thumb_width: int) -> None:
    original = cv2.imread(str(img_path))
    if original is None:
        print(f"[WARN] unreadable image: {img_path}")
        return

    image_out_dir = out_root / img_path.stem
    image_out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = image_out_dir / "debug_faces"
    debug_dir.mkdir(parents=True, exist_ok=True)

    faces = detect_faces(original, pad_ratio=face_pad)
    print(f"[INFO] {img_path.name}: detected {len(faces)} face(s)")

    version_images: List[np.ndarray] = []
    for i, params in enumerate(variants, start=1):
        result, debug_pairs = _process_single_variant(original, faces, params)
        out_path = image_out_dir / f"v{i:02d}.jpg"
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        version_images.append(result)

        heat, mad = _build_diff_map(original, result)
        if mad < MAD_TOO_WEAK_THRESHOLD:
            status = "TOO_WEAK"
        elif mad < MAD_OK_THRESHOLD:
            status = "OK"
        else:
            status = "STRONG"
        print(f"[DEBUG] {img_path.name} v{i:02d}: mad={mad:.3f} {status}")
        cv2.imwrite(str(debug_dir / f"v{i:02d}_diff_map.jpg"), heat, [cv2.IMWRITE_JPEG_QUALITY, 95])

        for face_idx, before, after in debug_pairs[:DEBUG_FACE_SAMPLES]:
            cv2.imwrite(str(debug_dir / f"v{i:02d}_face{face_idx:02d}_before.jpg"), before, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(debug_dir / f"v{i:02d}_face{face_idx:02d}_after.jpg"), after, [cv2.IMWRITE_JPEG_QUALITY, 95])
            pair = _make_before_after(before, after)
            cv2.imwrite(str(debug_dir / f"v{i:02d}_face{face_idx:02d}_before_after.jpg"), pair, [cv2.IMWRITE_JPEG_QUALITY, 95])

    build_contact_sheet(original, version_images, image_out_dir / "contact_sheet.jpg", thumb_width=thumb_width, face_boxes=faces)
    _write_params_csv(image_out_dir / "params.csv", variants, detector_count=len(faces))


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    images = iter_jpg_images(args.input_dir)
    if not images:
        print(f"[WARN] no jpg images found in: {args.input_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = build_variants()

    for img_path in images:
        process_image(img_path=img_path, out_root=args.output_dir, variants=variants, face_pad=args.face_pad, thumb_width=args.sheet_thumb_width)

    print(f"[DONE] Processed {len(images)} image(s). Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
