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

### CDF (raw + eq)
The CDF implementation will be released after the associated paper is accepted.

## License

```
This project is distributed under a custom non-commercial license. Any form of commercial use is prohibited.
```


