# Optimal Classification Trees with IDK Abstain Leaves
### Breast Cancer Diagnosis — Optimization Project

This project extends the **Optimal Classification Tree (OCT)** framework (Bertsimas & Dunn, 2017) by adding a third leaf option: **IDK (I Don't Know)**. Instead of forcing the model to classify every patient, ambiguous cases are abstained from and referred for further specialist testing.

---

## Motivation

In breast cancer diagnosis, misclassification carries very different costs:

| Error Type | Consequence | Estimated Cost |
|---|---|---|
| False Negative (FN) | Missed cancer → delayed treatment | ~$50,000 |
| False Positive (FP) | Unnecessary biopsy / treatment | ~$5,000 |
| IDK referral | Further specialist testing | ~$1,000 |

By allowing the model to abstain on genuinely ambiguous cases, we reduce costly errors at the expense of a small referral cost — a medically rational trade-off.

---

## Method

The OCT-IDK model is formulated as a **Mixed Integer Linear Program (MILP)** solved with [HiGHS](https://highs.dev/) via [JuMP.jl](https://jump.dev/).

Each leaf node in the decision tree has three mutually exclusive options:
1. Predict **Malignant**
2. Predict **Benign**
3. **IDK** — abstain and refer the patient

A penalty parameter **β** controls the cost of abstaining. The objective minimises:

$$\frac{1}{n} \sum_{t \in T_L} \left( L_{\text{err},t} + \beta \cdot q_t \right) + \alpha \sum_{t \in T_B} d_t$$

where $q_t$ is the number of patients routed to IDK leaf $t$. Higher β → fewer IDK leaves; lower β → more abstentions.

### Key design choices
- **Tree depth**: D = 3 (up to 8 leaves)
- **Feature selection**: Top 5 features by class-mean separation (Fisher-style score)
- **Sample size**: 150 patients (stratified subsample of Wisconsin Breast Cancer dataset)
- **Optimal β**: Selected at the elbow of the coverage–accuracy curve using the Kneedle algorithm

---

## Results (β = 0.2)

| Metric | Value |
|---|---|
| Coverage (classified) | 92.0% |
| Accuracy on classified | 99.3% |
| Sensitivity (malignant) | 98.0% |
| Specificity (benign) | 100% |
| IDK referred | 12 / 150 patients |
| False Negatives | 1 |
| False Positives | 0 |

**Estimated cost reduction vs Standard OCT: $48,000 (43.6%)**

---

## Visualisations

| File | Description |
|---|---|
| `oct_idk_tree.png` | Decision tree with IDK leaves at β = 0.2 |
| `oct_idk_pareto.png` | Coverage–accuracy Pareto frontier across β values |
| `oct_idk_sensitivity.png` | β sensitivity: IDK rate, coverage, and accuracy |
| `oct_idk_confusion.png` | Confusion matrix for classified patients |
| `oct_idk_metrics.png` | Performance metrics bar chart |
| `oct_idk_cost.png` | Cost comparison: Standard OCT vs OCT-IDK |

---

## Files

```
├── oct_idk.jl              # Full β sweep MILP solver (~40 min)
├── solve_demo.jl           # Fast single-β solve for β=0.2 (~5–10 min)
├── visualize_oct_idk.py    # Generates all 6 PNG visualisations
├── oct_idk_results.json    # Solver output (sweep + demo tree)
├── optiproj.ipynb          # Jupyter notebook with full analysis
└── breast_cancer.csv       # Wisconsin Breast Cancer dataset (place here)
```

---

## How to Run

### 1. Install dependencies

**Julia packages** (auto-installed on first run):
`JuMP`, `HiGHS`, `CSV`, `DataFrames`, `Statistics`, `JSON`

**Python packages**:
```bash
pip install matplotlib numpy
```

### 2. Run the solver

Full sweep across all β values (~40 min):
```bash
julia oct_idk.jl
```

Or just re-solve β = 0.2 and patch the JSON (~5–10 min):
```bash
julia solve_demo.jl
```

### 3. Generate visualisations

```bash
python3 visualize_oct_idk.py
```

---

## Dataset

**Wisconsin Breast Cancer Dataset** — 569 samples, 30 features (radius, texture, perimeter, area, smoothness, etc. — mean, SE, and worst).  
Source: UCI Machine Learning Repository.

Place `breast_cancer.csv` in the project root before running.

---

## References

- Bertsimas, D., & Dunn, J. (2017). *Optimal classification trees.* Machine Learning, 106(7), 1039–1082.
- Herbei, R., & Wegkamp, M. H. (2006). *Classification with reject option.* Canadian Journal of Statistics.
- Wisconsin Breast Cancer Dataset — UCI ML Repository.
