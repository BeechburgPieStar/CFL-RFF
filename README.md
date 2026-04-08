# Beyond Equalization: Confidence-Aware Fusion Learning for Robust RF Fingerprinting

## Overview
This repository provides a research-oriented implementation framework for **robust radio frequency fingerprinting (RFF)** under receiver and channel variations.

---

## Evaluation Protocol (Brief)

### Receiver Groups
Receivers are divided into four groups (R1–R4).  
A four-fold cross-validation protocol is adopted, where one group is used for testing and the remaining groups for training.

- **R1**: --test_round 0
- **R2**: --test_round 1
- **R3**: --test_round 2
- **RE**: --test_round 3
  
### Cross-Channel Settings
Three training configurations are considered:
- **Exp1**: train on one day (--train_date 1)
- **Exp2**: train on two days (--train_date 1 2)
- **Exp3**: train on three days (--train_date 1 2 3)

Performance is reported for each receiver group, and **Avg.** denotes the mean accuracy over R1–R4.

---

## Proposed Method and How to Run

### PatchNet (raw)
A single-branch PatchNet model using only raw IQ signals.

```
python main.py --train_date 1 2 3 --test_round 0
```

### PatchNet (eq)
A single-branch PatchNet model using only channel-equalized IQ signals.

```
python main.py --train_date 1 2 3 --test_round 0 --use_eq
```

These baselines are provided to analyze the respective roles of raw and equalized signal representations in cross-domain RFF.

### CFL (raw + eq)
CFL jointly leverages raw and equalized IQ signals via a **Dual-Branch Interactive Network (DBIN)** and an **Energy-based Dynamic Fusion (EDF)** module. EDF uses prediction energy as a per-sample confidence proxy to adaptively weight each branch's logits, supervised by a ranking regularization. The overall loss is:
```
L = CE(o_fused, y) + CE(o_raw, y) + CE(o_eq, y) + λ_rank · L_rank
```

```
python main.py --train_date 1 2 3 --test_round 0 --code_state train_test
```

#### Project Structure

```
CFL/
├── main.py              # Entry point: argument parsing, data preparation,
│                        #   training loop (History buffer, rank_loss, train/val/test)
├── backbones/
│   └── PatchNet.py      # DBIN: PatchEmbed (Conv1d) → MLP block → XInteract (CBI) → cls head
│                        #   Energy scores (conf_a, conf_b) computed here for EDF
├── utils/
│   └── load_data.py     # Loads paired (non_equalized, equalized) .pkl files,
│                        #   aligns classes, applies per-sample power normalization
├── dataset/
│   └── link.txt         # Download link for ManySig / ManyRx subsets of WiSig
│                        #   Expected layout: dataset/ManySig/{non_equalized,equalized}/date{1-4}/rx_*_data.pkl
├── weights/             # Auto-created. Checkpoints saved as:
│                        #   {dataset}_{exp}_date{dates}_round{r}_seed{s}_p{patch}_d{dim}_...pth
└── logs/                # Auto-created. No built-in logger; redirect stdout to persist:
                         #   python main.py ... 2>&1 | tee logs/run_date123_round0.log
```

#### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--train_date` | `[1, 2]` | Day indices used for training (Exp1: `1`, Exp2: `1 2`, Exp3: `1 2 3`) |
| `--test_round` | `1` | Receiver group held out for testing (0–3 → R1–R4) |
| `--code_state` | `only_test` | `only_train` / `only_test` / `train_test` |
| `--use_xi` | `1` | Enable (`1`) / disable (`0`) CBI cross-branch interaction |
| `--w_eq` | `1` | Enable the equalized branch |
| `--wo_eq` | `1` | Enable the raw branch |
| `--lamb_rank` | `1.0` | Weight λ for ranking regularization loss |

## License

```
This project is distributed under a custom non-commercial license. Any form of commercial use is prohibited.
```


