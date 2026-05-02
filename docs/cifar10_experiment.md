# CIFAR-10 Learning Rate Optimization — Experiment Documentation

## Overview

This document describes all methods implemented in `notebooks/cifar10.ipynb` for exploring adaptive learning rate strategies on CIFAR-10 image classification. The work compares a baseline Adam optimizer against three custom approaches: the Polling Method (from the base paper), a New LR Scheduler, and an Improved Polling method.

**Dataset:** CIFAR-10 (50 000 train / 10 000 test, 10 classes)  
**Model:** `SimpleCIFAR10CNN` — a small CNN (64→128→256 channels, two MaxPool layers, AdaptiveAvgPool + Linear head)  
**Base LR:** 1e-5 | **Weight decay:** 5e-4 | **Batch size:** 2048 | **Val split:** 10%

---

## Methods

### 1. Baseline

Standard Adam training with a fixed learning rate throughout all epochs.

- **Optimizer:** Adam (lr=1e-5, weight_decay=5e-4)
- **LR schedule:** None — static
- **Result (15 epochs):** val acc ~25.5%, test acc ~27.1%

---

### 2. Polling Method (base paper)

**Reference:** *Expediting Convergence via Polling Optimisation for Gradient Descent in Neural Networks* — Tan, Choong & Lau (2026).

**Core idea:** At every training batch, after computing gradients, try N candidate learning rates independently and keep only the weights produced by the LR that achieves the highest batch accuracy. This is an ensemble-inspired approach that dynamically selects the most effective LR at each gradient step without relying on any formula or schedule.

**Candidate LRs (relative to base lr=1e-5):**
```
1e-7, 1e-6, 1e-5, 1e-4, 1e-3
```

**Algorithm per batch:**
1. Forward pass → loss → backward (compute gradients)
2. For each candidate LR: clone model/optim state, apply step, measure batch accuracy
3. Keep the state (weights + optim) from the best-accuracy candidate

**Result (15 epochs):** val acc ~31.1% — notably better than baseline with the same number of epochs. The mean chosen LR per epoch decreases over time as the model approaches a better region (exploration → exploitation).

---

### 3. New LR Scheduler

**Core idea:** Adjust the learning rate automatically based on the ratio of how much the loss improved over a patience window. Uses logarithmic formulas so the factor passes through (1, 1) — i.e., no change when loss ratio is exactly 1 — and scales smoothly without risk of gradient explosion.

**Parameters:** patience=30, beta=0.9 (EMA smoothing), no lr bounds

**Algorithm:**
- Track loss with EMA (`ema = beta * ema + (1-beta) * loss`)
- Every `patience` steps: compute `ratio = first_loss / ema`
- Alternate direction (up/down) each time the model stops improving

**LR update formulas:**

When loss is improving (`ratio > 1`):
- Increase direction: `lr *= 1 + log(ratio)` — factor > 1, grows with improvement
- Decrease direction: `lr *= 1 / (1 + log(ratio))` — factor < 1 (errata correction)

When loss is not improving (`ratio ≤ 1`):
- Uses `2*(1 - 2^(-ratio))` / mirror formula — bounded, passes through (1,1)

See `images/learning_rate_factor_formula.png`, `images/learning_rate_factor_formula_with_log.png`, and `images/all_learning_rate_factor_formula.png` for formula visualizations.

**Errata:** The original decrease formula was written as `1 + log(1/x) = 1 - log(x)`, which can go negative for large ratios. The correct formula is `1 / (1 + log(x))`.

**Result (15 epochs):** val acc ~26.7%, **test acc ~55.9%** — the test result is from a longer/later run saved to disk, suggesting the scheduler works well given enough training time.

---

### 4. Improved Polling Method

**Core idea:** A middle ground between pure polling (tries N fixed candidates) and pure formula-based scheduling (no exploration). Start with the current LR, compute a loss improvement ratio over a patience window, derive one "increase" factor and one "decrease" factor from that ratio, then test three candidates: `lr * factor_up`, `lr`, `lr * factor_down`. Pick the one that reduces loss the most.

This differs from the base paper's polling in that:
- The candidate set is derived dynamically from the loss signal (not predetermined)
- The factors scale with how much the model has improved, so steps are larger early and smaller late

**Parameters:** patience (configurable), min_lr, max_lr

**LR factor formulas (same as New LR Scheduler):**
- `factor_up = 1 + log(ratio)` where `ratio = window_start_loss / current_loss`
- `factor_down = 1 / (1 + log(ratio))`

**Algorithm per batch:**
- Accumulate batches for `patience` steps with normal gradient steps
- On the `patience`-th step: try three candidate LRs, pick the best by lowest loss on the batch

**Bugs found and fixed (v2 → v3):**

| Bug | Description | Impact |
|-----|-------------|--------|
| **Missing `optim.step()`** | Non-polling batches returned the pre-step snapshot without ever calling `optim.step()`. Gradients were computed but discarded every batch except the polling batch. | Model barely learned — loss stuck at ~2.303 (random), accuracy ~10% |
| **`first_loss` never resets** | `patience_counter` was incremented at the end, so after polling reset it would be 1, not 0, on the next batch. `first_loss` was set only when counter==0, so it only recorded the very first batch ever. | Polling always compared against the initial loss, making the ratio meaningless after the first window |
| **Wrong decrease formula** | Used `1 - log(x)` instead of the corrected `1 / (1 + log(x))` | For large improvement ratios, factor could go negative, making the LR negative |

---

## Results Comparison (15 epochs)

| Method | Val Acc | Notes |
|--------|---------|-------|
| Baseline | ~25.5% | Fixed lr=1e-5 |
| Polling (paper) | ~31.1% | Best among 15-epoch runs |
| New LR Scheduler | ~26.7% | Test acc ~55.9% in extended run |
| Improved Polling (v2, broken) | ~10.5% | Stuck at random due to bugs |
| Improved Polling (v3, fixed) | TBD | Bugs corrected, ready to re-run |

---

## Architecture

```
SimpleCIFAR10CNN
├── Conv2d(3, 64, 3, pad=1) + ReLU
├── Conv2d(64, 64, 3, pad=1) + ReLU
├── MaxPool2d(2)
├── Conv2d(64, 128, 3, pad=1) + ReLU
├── Conv2d(128, 128, 3, pad=1) + ReLU
├── MaxPool2d(2)
├── Conv2d(128, 256, 3, pad=1) + ReLU
├── AdaptiveAvgPool2d((1, 1))
├── Flatten
└── Linear(256, 10)
```

---

## Normalization

Per-channel mean/std computed from the training set:
- Mean (R, G, B): [125.31, 122.95, 113.87]
- Std  (R, G, B): [62.99, 62.09, 66.70]
