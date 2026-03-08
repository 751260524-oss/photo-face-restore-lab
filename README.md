# AI Old Photo Restore System

## Folder structure

```text
photo-face-restore-lab
├─ .github/workflows/run_face_restore.yml
├─ scripts/restore_system.py
├─ input/
├─ output/
├─ third_party/
└─ .gitignore
```

## How to use

1. Put old photos into `input/`
2. Push to GitHub
3. Open **Actions**
4. Run **AI Old Photo Restore System**
5. Download `restore-output`

## Output structure

```text
output/
├─ 1/
├─ 2/
├─ comparisons/
├─ final_results/
├─ logs/
├─ SUMMARY.md
└─ summary.json
```

- `final_results/`: final restored images for direct use
- `comparisons/`: before/after comparison images
- `logs/`: per-image error logs when a file fails
- `SUMMARY.md`: human-readable summary
- `summary.json`: machine-readable summary
