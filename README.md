# Optimal Classification Trees with IDK Abstain Leaves
### Breast Cancer Diagnosis — Optimisation Project

This project extends the **Optimal Classification Tree (OCT)** framework (Bertsimas & Dunn, 2017) by introducing a third leaf option: **IDK (I Don't Know)**. Instead of forcing every patient into a prediction, the model can abstain on ambiguous cases and refer them for further specialist testing — a medically rational design that reduces costly misdiagnoses.

---

## 1. MILP Formulation

### Sets and Indices

| Symbol | Meaning |
|---|---|
| $i \in \{1, \ldots, n\}$ | Data points (patients) |
| $t \in T$ | Tree nodes |
| $j \in \{1, \ldots, p\}$ | Features |
| $k \in \{\text{Benign, Malignant, IDK}\}$ | Prediction classes |

### Decision Variables

| Variable | Type | Meaning |
|---|---|---|
| $a_{jt}$ | $\{0,1\}$ | 1 if feature $j$ is used to split at node $t$ |
| $b_t$ | $\mathbb{R}$ | Threshold value at branch node $t$ |
| $z_{it}$ | $\{0,1\}$ | 1 if sample $i$ is routed to leaf $t$ |
| $c_{kt}$ | $\{0,1\}$ | 1 if leaf $t$ predicts class $k$ |
| $q_t$ | $\geq 0$ | Number of samples assigned to IDK leaf $t$ |

### Objective Function

$$\min \; \frac{1}{n} \sum_{t \in \text{leaves}} L_{\text{err}}(t) \;+\; \beta \sum_{t \in \text{leaves}} q_t \;+\; \alpha \,|\text{splits}|$$

- **Classification error** — misclassifications on committed leaves
- **IDK penalty** — $\beta \cdot q_t$ penalises abstaining; higher $\beta$ forces more classifications
- **Tree complexity** — $\alpha$ regularises the number of splits

The key novelty is the $q_t$ term: it introduces abstention into the optimisation, allowing the model to defer genuinely uncertain cases instead of forcing a prediction.

---

## 2. Constraints

### Flow Constraints
Each patient reaches exactly one leaf:

$$\sum_{t \in \text{leaves}} z_{it} = 1 \quad \forall \, i$$

### Split Constraints
Ensure valid branching based on feature thresholds:

$$\sum_j a_{jt} = d_t \quad \forall \, t \qquad d_t \leq d_{\text{parent}(t)}$$

### Class Assignment Constraints
Each active leaf either predicts exactly one class **or** abstains — never both:

$$\sum_{k \in \{\text{Benign, Malignant}\}} c_{kt} + \text{idk}_t = 1$$

- $\text{idk}_t = 1$: leaf abstains (no prediction)
- $\text{idk}_t = 0$: leaf predicts exactly one class

### IDK Definition
$q_t$ counts patients in IDK leaves. Since $q_t = \sum_i z_{it} \times \text{idk}_t$ is nonlinear, it is **linearised via McCormick envelope**:

$$q_t \leq n \cdot \text{idk}_t$$
$$q_t \leq \sum_i z_{it}$$
$$q_t \geq \sum_i z_{it} - n(1 - \text{idk}_t)$$

### Error Counting
Only penalise classified leaves (IDK leaves incur no misclassification error):

$$L_{\text{err}}(t) \geq N_t - N_{kt} - M(1 - c_{kt}) \quad k \in \{1, 2\}$$

> Unlike standard OCT, the binary $\text{idk}_t$ variable modifies both class assignment and error counting — IDK leaves contribute zero classification error but pay the $\beta$ abstention penalty.

---

## 3. Results

### Decision Tree — Depth 3, β = 0.2

The tree makes splits on the five most discriminative features (selected by Fisher-style class-mean separation score). Two leaves output **IDK**, routing ambiguous patients to further testing.

```
                      concave pts mean ≤ 0.056
                      /                        \
           area worst ≤ 876.5          concave pts worst ≤ 0.137
           /           \                /                   \
  radius worst ≤ 16.5  texture worst ≤ 25.1   perimeter worst ≤ 105.0   area worst ≤ 1200.0
   /      \              /       \               /         \               /          \
Benign   Malignant    IDK    Malignant        Benign       IDK          Benign    Malignant
(n=70)   (n=10)     (n=7)    (n=8)           (n=4)       (n=5)         (n=9)     (n=37)
```

**Key properties of β:**
- Lower β → more IDK decisions (conservative behaviour)
- Higher β → fewer IDK leaves (converges to standard OCT)
- Tree structure and IDK decisions are **jointly optimised** via MILP
- Model evaluated across β ∈ {0.1, 0.2, 0.3, 0.5, 0.8}

### Performance Metrics at β = 0.2

| Metric | Value |
|---|---|
| Coverage (classified) | **92.0%** |
| Accuracy (classified) | **99.3%** |
| Accuracy (all patients) | **91.3%** |
| Sensitivity (malignant) | **98.0%** |
| Specificity (benign) | **100.0%** |
| IDK referred | 12 / 150 patients |

### Confusion Matrix (classified patients only)

|  | Predicted Malignant | Predicted Benign |
|---|---|---|
| **Actual Malignant** | TP = 49 (35.5%) | FN = 1 (0.7%) |
| **Actual Benign** | FP = 0 (0.0%) | TN = 88 (63.8%) |

> 12 patients referred (IDK) are excluded from the confusion matrix — they are routed to specialist testing rather than classified.

### β Sensitivity Analysis

| β | IDK patients | Coverage | Accuracy (classified) |
|---|---|---|---|
| 0.1 | 18 | 88.0% | 100.0% |
| **0.2** | **12** | **92.0%** | **99.3%** |
| 0.3 | 0 | 100.0% | 97.3% |
| 0.5 | 0 | 100.0% | 97.3% |
| 0.8 | 0 | 100.0% | 97.3% |

β = 0.2 is selected as the **elbow point**: it is the highest penalty at which the model still meaningfully abstains, achieving near-perfect precision on classified patients while deferring only the genuinely ambiguous cases.

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

### Generated Visualisations

| File | Description |
|---|---|
| `oct_idk_tree.png` | Full depth-3 decision tree with IDK leaves at β = 0.2 |
| `oct_idk_pareto.png` | Coverage–accuracy Pareto frontier across β values |
| `oct_idk_sensitivity.png` | β sensitivity: IDK rate, coverage, and accuracy |
| `oct_idk_confusion.png` | Confusion matrix for classified patients only |
| `oct_idk_metrics.png` | Performance metrics bar chart |
| `oct_idk_cost.png` | Estimated cost reduction: Standard OCT vs OCT-IDK |

---

## How to Run

**Julia packages** (auto-installed on first run): `JuMP`, `HiGHS`, `CSV`, `DataFrames`, `Statistics`, `JSON`

**Python packages**: `matplotlib`, `numpy`

```bash
# Full β sweep (~40 min)
julia oct_idk.jl

# Fast re-solve for β=0.2 only (~5–10 min)
julia solve_demo.jl

# Generate all visualisations
python3 visualize_oct_idk.py
```

---

## Dataset

**Wisconsin Breast Cancer Dataset** — 569 samples, 30 features (radius, texture, perimeter, area, smoothness etc. — mean, SE, and worst). Source: UCI Machine Learning Repository.

Place `breast_cancer.csv` in the project root before running.

---

## References

- Bertsimas, D., & Dunn, J. (2017). *Optimal classification trees.* Machine Learning, 106(7), 1039–1082.
- Herbei, R., & Wegkamp, M. H. (2006). *Classification with reject option.* Canadian Journal of Statistics.
- Wisconsin Breast Cancer Dataset — UCI ML Repository.
