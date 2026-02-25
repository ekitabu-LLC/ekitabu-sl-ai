# KSL Model Migration Guide

## Quick Summary

Your current Modal account ran out of credits. This guide helps you migrate to a new Modal account.

## What You Have Locally (Already Safe)

These files are on your Mac and don't need downloading:

| Directory | Size | Contents |
|-----------|------|----------|
| `*.py` files | ~500KB | All training scripts (v4-v9) |
| `dataset/` | 2.7GB | Raw video files |
| `validation-data/` | 3.0GB | Validation videos |
| `dataset/processed/` | 206MB | Local processed features |
| `dataset/lstm_processed/` | 26MB | LSTM features |
| `docs/plans/` | - | Implementation plans |

## What's on Modal (NEEDS DOWNLOADING)

These are stored in Modal's cloud and will be lost without action:

| Directory | Contents | Priority |
|-----------|----------|----------|
| `/data/train_v2/` | Preprocessed training features | **CRITICAL** |
| `/data/val_v2/` | Preprocessed validation features | **CRITICAL** |
| `/data/checkpoints_*/` | Saved model weights | Optional |

---

## Step 1: Download from Old Modal Account (DO THIS NOW!)

Before your credits fully expire, run:

```bash
cd /Users/hassan/Documents/Hassan/ksl-dir-2
modal run download_modal_data.py
```

This will create a `modal_backup/` folder with all your processed features.

**If this fails due to zero credits:** You can regenerate the features from raw videos, but it takes ~2 hours of GPU time.

---

## Step 2: Create New Modal Account

1. Go to https://modal.com
2. Sign up with a **different email**
3. New accounts get $30 free credits

---

## Step 3: Setup New Modal CLI

```bash
# Logout from old account
modal token delete

# Login to new account
modal token new
# This opens a browser - login with your new account
```

Verify you're on the new account:
```bash
modal profile current
```

---

## Step 4: Upload Data to New Account

```bash
modal run upload_to_new_modal.py
```

This creates a new `ksl-dataset-vol` volume in your new account and uploads all the backup data.

---

## Step 5: Resume Training

Once data is uploaded:

```bash
# Run the latest training version
modal run train_ksl_v9.py
```

---

## Alternative: Regenerate Features (If Download Fails)

If you couldn't download the Modal data, you can regenerate features from raw videos:

1. Upload raw videos to new Modal volume:
```bash
modal run upload_raw_videos.py  # Need to create this
```

2. Run feature extraction:
```bash
modal run extract_features_v2.py
```

This takes ~2 hours but rebuilds everything from scratch.

---

## Project Status Summary

### Model Performance (v8 - Last Successful)
- Overall: 83.65%
- Problematic: 444 at 0%, 89 at 56%
- Working well: Words at 88.6%, most numbers at 80%+

### v9 Goals (In Progress)
- Fix 444 confusion with temporal features
- Target: 444 > 40%, Overall > 85%

### Key Files for Training
- `train_ksl_v9.py` - Latest training script
- `assess_v9.py` - Evaluation script
- `analyze_444_vs_54.py` - Analysis tool

---

## Checklist

- [ ] Run `modal run download_modal_data.py` (old account)
- [ ] Create new Modal account
- [ ] Run `modal token new` (authenticate new account)
- [ ] Run `modal run upload_to_new_modal.py`
- [ ] Verify with `modal volume ls ksl-dataset-vol`
- [ ] Resume training with `modal run train_ksl_v9.py`
