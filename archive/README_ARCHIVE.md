# Archive Directory

This directory contains old, experimental, or broken code that has been superseded by the minimal training approach.

**Date Archived:** 2025-12-09

---

## Contents

### `old_training/` - Previous Training Modules

**Reason:** Replaced by `train_minimal.py` (all-in-one script)

Files:
- `train.py` - Original modular training orchestrator (600 lines)
- `model_builder.py` - Autoencoder architecture builder
- `threshold_finder.py` - 5 threshold optimization methods
- `evaluator.py` - Model evaluation and visualization

**Note:** These were part of a modular design that was later consolidated for simplicity.

---

### `old_src/` - Broken/Incomplete Source Files

**Reason:** Feature preprocessing mismatch issues

Files:
- `inference.py` - Original batch inference script
- `inference_minimal.py` - Attempted multi-layer inference (fails on feature count mismatch)
- `test_minimal.py` - Testing script (fails on feature alignment)
- `realtime_detector.py` - Production template (incomplete, complex)

**Issue:** These files fail because:
1. Training creates different feature counts than inference
2. Categorical encoding is inconsistent
3. No feature metadata saved during training

**Status:** Needs rewrite with feature alignment logic

---

### `old_docs/` - Previous Documentation

**Reason:** Superseded by focused minimal guides

Files:
- `MULTI_LAYER_IDS_DEPLOYMENT_GUIDE.md` (2,966 lines) - Comprehensive but too detailed
- `TRAINING_OPTIMIZATION_PLAN.md` - Planning document (implemented)
- `REFACTORING_SUMMARY.md` - Old refactoring log
- `REALTIME_DEPLOYMENT.md` - Real-time deployment guide (outdated)

**Note:** Contains useful reference material but no longer maintained.

---

### `utilities/` - Helper Scripts

**Reason:** One-time use, no longer needed

Files:
- `get_raw_samples.py` - Extract raw samples from dataset
- `setup_structure.py` - Initial project structure setup

---

## Should I Use Archived Code?

**NO** - Use the current minimal system instead:
- `src/train_minimal.py` - Training
- `src/data_loader.py` - Data loading
- `src/preprocessor.py` - Preprocessing
- `src/realtime_simple.py` - Real-time demo

**Only use archive for:**
- Understanding previous approaches
- Extracting specific utility functions
- Troubleshooting historical issues

---

## Restoration

If you need to restore archived files:

```bash
# Example: Restore old training modules
cp archive/old_training/model_builder.py src/
cp archive/old_training/threshold_finder.py src/
cp archive/old_training/evaluator.py src/
```

---

## Archive Policy

Files are archived (not deleted) when:
1. Superseded by better implementation
2. No longer functional due to design changes
3. Experimental/incomplete features
4. Redundant with current codebase

Files should be **deleted** (not archived) when:
- Contain sensitive information
- Generated outputs (logs, temp files)
- Dependencies/packages (.venv, node_modules)

---

**Last Updated:** 2025-12-09
