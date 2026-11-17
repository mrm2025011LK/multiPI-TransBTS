# Results

Run: Tesla T4, 5 epochs, batch 12, subset of 80 patients (4348 training slices), 30 unseen patients, seed 0. Training time 19 min, peak GPU memory 7.8 GB.

## Baseline (released settings)

### Official protocol: official test slices of the subset (520 slices; 79 of 79 test patients also occur in training)

| Region | Dice | Sensitivity | Precision | released `hausdorff_95` | spatial HD95 (px) |
|---|---|---|---|---|---|
| WT | 0.903 | 0.937 | 0.872 | 2.811 | 13.75 |
| TC | 0.887 | 0.903 | 0.872 | 1.987 | 6.18 |
| ET | 0.786 | 0.706 | 0.886 | 1.768 | 6.99 |
| Mean | 0.859 | 0.848 | 0.877 | 2.189 | 8.97 |

### Same weights on unseen patients (30 patients, 1874 slices, none in training)

| Region | Dice | Sensitivity | Precision | released `hausdorff_95` | spatial HD95 (px) |
|---|---|---|---|---|---|
| WT | 0.881 | 0.909 | 0.855 | 2.945 | 17.62 |
| TC | 0.806 | 0.798 | 0.814 | 2.266 | 11.70 |
| ET | 0.747 | 0.637 | 0.903 | 1.863 | 10.11 |
| Mean | 0.811 | 0.782 | 0.857 | 2.358 | 13.14 |

Per-patient Dice on unseen patients (mean ± std over patients): WT 0.861 ± 0.068, TC 0.777 ± 0.197, ET 0.577 ± 0.303

## Proposed change: ET loss with beta = 0.7 (baseline beta = 0.5)

Evaluated on the unseen set, same split, same seed.

| Model | Region | Dice | Sensitivity | Precision | per-patient Dice | spatial HD95 (px) |
|---|---|---|---|---|---|---|
| baseline | WT | 0.881 | 0.909 | 0.855 | 0.861 | 17.62 |
| baseline | TC | 0.806 | 0.798 | 0.814 | 0.777 | 11.70 |
| baseline | ET | 0.747 | 0.637 | 0.903 | 0.577 | 10.11 |
| variant | WT | 0.882 | 0.888 | 0.876 | 0.863 | 15.54 |
| variant | TC | 0.799 | 0.762 | 0.841 | 0.771 | 11.43 |
| variant | ET | 0.802 | 0.779 | 0.827 | 0.610 | 10.44 |

