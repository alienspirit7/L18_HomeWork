# Product Requirements Document (PRD)
## Logistic Regression Implementation from Scratch

---

## Document Information

| Field | Value |
|-------|-------|
| **Project Name** | Logistic Regression with Gradient Descent |
| **Version** | 1.1 |
| **Date** | 2025-11-15 |
| **Development Time** | 2 hours |
| **Author** | Development Team |
| **Status** | Complete |

---

## 1. Executive Summary

### 1.1 Project Overview
Develop a complete implementation of logistic regression using gradient descent from scratch in Python, demonstrating the mathematical foundations of binary classification with clear visualizations and comprehensive documentation.

### 1.2 Objectives
- Implement logistic regression without using ML libraries (scikit-learn, etc.)
- Demonstrate gradient descent optimization visually
- Create educational materials showing the algorithm's inner workings
- Provide reproducible, well-documented code
- Validate manual results against a scikit-learn baseline for parity

### 1.3 Success Criteria
- ✓ Accurate implementation of sigmoid, gradient, and update formulas
- ✓ Model converges to >99% accuracy
- ✓ All visualizations clearly show decision boundaries
- ✓ Complete documentation with code references
- ✓ Reproducible results with fixed random seed
- ✓ Comparison script shows manual vs scikit-learn agreement (accuracy + coefficients)

---

## 2. Scope

### 2.1 In Scope
1. **Data Generation**
   - Random point generation with two separable clusters
   - Configurable sample size and cluster parameters
   - CSV export functionality

2. **Gradient Descent Implementation**
   - Sigmoid activation function
   - Log-likelihood calculation
   - Gradient computation
   - Coefficient update mechanism
   - Early stopping criteria

3. **Visualizations**
   - Initial data scatter plot (2D)
   - 3D sigmoid surface with decision boundary
   - Training metrics (error and log-likelihood over time)

4. **Output Artifacts**
   - Training history (all iterations)
   - Final predictions with error metrics
   - Multiple visualization formats

5. **Documentation**
   - Comprehensive README with examples
   - Code comments and docstrings
   - Development history (prompts.md)

6. **Model Validation & Comparison**
   - Scikit-learn baseline using `LogisticRegression`
   - CSV summary of manual vs library metrics
   - Visual comparisons (coefficients, probabilities, decision surfaces)

### 2.2 Out of Scope
- Multi-class classification (only binary)
- Feature engineering or preprocessing
- Cross-validation or train/test splits
- Regularization (L1/L2)
- Web interface or API
- Model persistence/serialization
- Automated deployment or packaging

---

## 3. Technical Requirements

### 3.1 Mathematical Formulas

**Formula 1: Sigmoid Function**
```
p_i = 1 / (1 + exp(-(β₀x₀ᵢ + β₁x₁ᵢ + β₂x₂ᵢ)))
```
- Must handle numerical stability (prevent overflow/underflow)
- Input: weighted sum z
- Output: probability in range [0, 1]

**Formula 2: Gradient Calculation**
```
g = X^T · (y - p)
```
- Vectorized implementation required
- Computes partial derivatives for all coefficients simultaneously

**Formula 3: Coefficient Update**
```
β^(k+1) = β^(k) + η·g
```
- Learning rate η = 0.1 (default)
- Gradient ascent (maximizing log-likelihood)

### 3.2 Algorithm Specifications

**Data Generation:**
- Sample size: 4,000 points (2,000 per class)
- Distribution: Gaussian/Normal
- Group 0: μ = (0.2, 0.2), σ = 0.08
- Group 1: μ = (0.8, 0.8), σ = 0.08
- Feature range: [0, 1] (clipped)

**Training Parameters:**
- Initial coefficients: β = [0, 0, 0]
- Learning rate: η = 0.1
- Max iterations: 10,000
- Convergence threshold: avg_error < 0.01
- Random seed: 42 (reproducibility)

**Performance Targets:**
- Accuracy: ≥ 99%
- Convergence: < 500 iterations (Actual: 6 iterations!)
- Average error: < 0.01
- Execution time: < 5 seconds

### 3.3 Technology Stack

**Required Libraries:**
- `numpy` ≥ 1.19.0 - Numerical computing
- `pandas` ≥ 1.1.0 - Data manipulation
- `matplotlib` ≥ 3.3.0 - Visualization (including 3D)
- `scikit-learn` ≥ 1.0.0 - Baseline logistic regression for validation

**Python Version:**
- Python 3.7 or higher

**Development Environment:**
- Cross-platform (macOS, Linux, Windows)
- No GPU required
- Minimal dependencies

---

## 4. Functional Requirements

### 4.1 Data Generation Module (`generate_points.py`)

**FR-1.1: Point Generation**
- System SHALL generate random points using Gaussian distribution
- System SHALL support configurable sample size
- System SHALL ensure two separable clusters

**FR-1.2: Data Export**
- System SHALL save data to CSV format
- System SHALL include columns: x1, x2, R
- System SHALL validate data integrity

**FR-1.3: Visualization**
- System SHALL create 2D scatter plot
- System SHALL use color coding (blue=0, red=1)
- System SHALL save plot as PNG

### 4.2 Logistic Regression Module (`logistic_regression.py`)

**FR-2.1: Model Initialization**
- System SHALL initialize coefficients to zero
- System SHALL add bias term (x₀ = 1)
- System SHALL set hyperparameters

**FR-2.2: Training Loop**
- System SHALL implement iterative gradient descent
- System SHALL compute sigmoid probabilities
- System SHALL calculate gradient vector
- System SHALL update coefficients
- System SHALL track convergence metrics

**FR-2.3: Early Stopping**
- System SHALL monitor average error
- System SHALL stop when error < threshold
- System SHALL support max iteration limit

**FR-2.4: Output Generation**
- System SHALL save iteration history to CSV
- System SHALL save final predictions to CSV
- System SHALL include average error calculation

**FR-2.5: 3D Visualization**
- System SHALL create 3D scatter plot
- System SHALL plot sigmoid surface
- System SHALL show decision boundary at z=0.5
- System SHALL use color for actual class
- System SHALL use shape for predicted class

**FR-2.6: Training Metrics Visualization**
- System SHALL plot error vs iterations
- System SHALL plot log-likelihood vs iterations
- System SHALL use dual y-axes
- System SHALL save as single image

### 4.3 Pipeline Module (`main.py`)

**FR-3.1: Orchestration**
- System SHALL run data generation first
- System SHALL verify data exists before training
- System SHALL handle script execution errors

**FR-3.2: User Feedback**
- System SHALL display progress messages
- System SHALL show success/failure status
- System SHALL summarize generated files

### 4.4 Comparison Module (`logistic_regression_sklearn.py`)

**FR-4.1: Baseline Training**
- System SHALL load the generated dataset
- System SHALL train scikit-learn's `LogisticRegression` with `lbfgs`
- System SHALL report coefficients, accuracy, iterations

**FR-4.2: Manual Result Ingestion**
- System SHALL load manual history/results CSV files
- System SHALL extract final coefficients and probabilities
- System SHALL compute manual accuracy and misclassification count

**FR-4.3: Metrics Comparison**
- System SHALL build a consolidated table (manual vs scikit-learn vs difference)
- System SHALL save the comparison to CSV
- System SHALL display the table in terminal output

**FR-4.4: Visualization**
- System SHALL produce bar charts for coefficients
- System SHALL scatter predicted probabilities with correlation annotation
- System SHALL chart accuracy/iteration metrics side-by-side
- System SHALL render a decision-surface comparison figure (manual vs scikit-learn)

---

## 5. Non-Functional Requirements

### 5.1 Performance
- **NFR-1**: Training SHALL complete within 5 seconds
- **NFR-2**: Convergence SHALL occur within 500 iterations
- **NFR-3**: Memory usage SHALL be < 500 MB

### 5.2 Usability
- **NFR-4**: Single command execution via `main.py`
- **NFR-5**: Clear terminal output with progress indicators
- **NFR-6**: Comprehensive README with examples

### 5.3 Reliability
- **NFR-7**: Reproducible results with fixed random seed
- **NFR-8**: Numerical stability (no overflow/underflow)
- **NFR-9**: Graceful error handling

### 5.4 Maintainability
- **NFR-10**: Modular code structure (separate scripts)
- **NFR-11**: Clear function names and comments
- **NFR-12**: Version control with .gitignore

### 5.5 Documentation
- **NFR-13**: README with formula references
- **NFR-14**: Code location pointers (file:line)
- **NFR-15**: Development history tracking

---

## 6. User Stories

### 6.1 Primary User: Student/Educator

**US-1: Understanding Logistic Regression**
> As a student learning machine learning,
> I want to see how logistic regression works step-by-step,
> So that I can understand the mathematical foundations.

**Acceptance Criteria:**
- ✓ Formula implementations are clearly visible in code
- ✓ Intermediate values are tracked and saved
- ✓ Visualizations show the decision boundary formation

**US-2: Experimenting with Parameters**
> As a student,
> I want to modify learning rate and sample size,
> So that I can see how they affect convergence.

**Acceptance Criteria:**
- ✓ Parameters are configurable at top of scripts
- ✓ Changes don't require deep code modifications
- ✓ Results are reproducible

**US-3: Presenting Results**
> As an educator,
> I want high-quality visualizations and clear documentation,
> So that I can use this for teaching purposes.

**Acceptance Criteria:**
- ✓ Professional-quality plots (high DPI)
- ✓ Clear labels and legends
- ✓ Comprehensive README

---

## 7. System Architecture

### 7.1 Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                        main.py                          │
│                 (pipeline orchestrator)                 │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
               │                      │
               ▼                      ▼
     ┌─────────────────────┐  ┌──────────────────────────┐
     │ generate_points.py  │  │ logistic_regression.py   │
     │ (data build + 2D    │  │ (manual training +       │
     │ scatter plot)       │  │ visualization)           │
     └──────────┬──────────┘  └──────────┬───────────────┘
                │                        │
                ▼                        ▼
      random_points.csv          gradient history, metrics,
      + points_visualization     predictions, manual plots
                │                        │
                └────────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   results_output/    │
                  │  • datasets          │
                  │  • histories         │
                  │  • manual plots      │
                  │  • comparison files  │
                  └──────────┬───────────┘
                             │
                             ▼
               ┌──────────────────────────────┐
               │ logistic_regression_sklearn  │
               │ .py (baseline comparison)    │
               └──────────────────────────────┘
```

### 7.2 Data Flow

```
Random Seed (42)
    │
    ▼
Generate Points → CSV File
    │                 │
    │                 ▼
    │            Load Dataset
    │                 │
    │                 ▼
    │         Add Bias Term (x₀=1)
    │                 │
    │                 ▼
    │         Initialize β=[0,0,0]
    │                 │
    │                 ▼
    │         ┌───────────────┐
    │         │ Training Loop │◄──┐
    │         │  • Sigmoid    │   │
    │         │  • Gradient   │   │
    │         │  • Update β   │   │
    │         └───────┬───────┘   │
    │                 │           │
    │                 ▼           │
    │         Check Convergence   │
    │                 │           │
    │            Yes  │  No       │
    │         ┌───────┴───────────┘
    │         │
    │         ▼
    │    Final Predictions
    │         │
    │         ├────► Visualizations & Reports
    │         │
    │         └────► Load manual metrics
    │                      │
    │                      ▼
    │             Train scikit-learn baseline
    │                      │
    │                      ▼
    └──────────────► Combined comparison outputs
```

---

## 8. File Specifications

### 8.1 Output Files

| File | Format | Size | Contents |
|------|--------|------|----------|
| `random_points.csv` | CSV | ~120 KB | x1, x2, R (4,000 rows) |
| `gradient_descent_history.csv` | CSV | ~25 KB | iteration, β₀, β₁, β₂, log_likelihood, avg_error |
| `logistic_regression_results.csv` | CSV | ~180 KB | x1, x2, R, σ(x), error + avg |
| `points_visualization.png` | PNG | ~200 KB | 800×800px, 2D scatter |
| `classification_results_3d.png` | PNG | ~800 KB | 1400×1000px, 3D surface |
| `training_metrics.png` | PNG | ~150 KB | 1200×600px, dual-axis |
| `comparison_manual_vs_sklearn.csv` | CSV | ~10 KB | Manual vs scikit-learn coefficients, accuracy, iterations |
| `comparison_visualization.png` | PNG | ~400 KB | 3-panel chart (coefficients, probability scatter, metrics) |
| `comparison_decision_surfaces.png` | PNG | ~850 KB | Side-by-side 3D decision surfaces (manual vs scikit-learn) |

### 8.2 Code Files

| File | Lines | Purpose |
|------|-------|---------|
| `generate_points.py` | 69 | Data generation |
| `logistic_regression.py` | 255 | Manual training & visualization |
| `logistic_regression_sklearn.py` | 329 | Scikit-learn baseline + comparisons |
| `main.py` | 89 | Pipeline orchestrator |
| `requirements.txt` | 3 | Dependencies |
| `.gitignore` | ~45 | Git exclusions |
| `README.md` | 734 | Documentation & tutorial |
| `prompts.md` | ~105 | Development history |
| `PRD.md` | This file | Requirements |

---

## 9. Testing Requirements

### 9.1 Unit Testing
- Sigmoid function outputs [0, 1]
- Gradient calculation dimensions match
- Coefficient updates are applied correctly

### 9.2 Integration Testing
- Data generation → Training pipeline works
- CSV files are readable by training script
- Comparison script consumes manual CSV outputs
- All visualizations (manual + comparison) are generated

### 9.3 Validation Testing
- Model accuracy ≥ 99%
- Convergence occurs < 500 iterations
- No errors with seed=42
- Manual vs scikit-learn accuracy delta ≤ 0.01%

---

## 10. Constraints and Assumptions

### 10.1 Constraints
- **Time**: 2-hour development window
- **Scope**: Binary classification only
- **Dependencies**: numpy, pandas, matplotlib, scikit-learn
- **Platform**: Must work on macOS, Linux, Windows

### 10.2 Assumptions
- Users have Python 3.7+ installed
- Users can install packages via pip
- Data is linearly separable
- No missing values or outliers
- Fixed random seed ensures reproducibility

---

## 11. Risk Management

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Numerical overflow in exp() | Medium | High | Use np.clip() for stability |
| Non-convergence | Low | Medium | Set max iterations limit |
| Dependency issues | Low | Low | Pin versions in requirements.txt |
| Platform incompatibility | Low | Medium | Test on multiple OS |
| Poor visualization quality | Low | Low | Use high DPI (150) |

---

## 12. Success Metrics

### 12.1 Technical Metrics
- [x] Accuracy: 100% (Target: ≥99%) ✓ **Perfect Score!**
- [x] Convergence: 6 iterations (Target: <500) ✓ **Far Exceeded**
- [x] Error: 0.000168 (Target: <0.01) ✓ **Exceeded**
- [x] Execution time: <0.1s (Target: <5s) ✓ **Exceeded**
- [x] Scikit-learn baseline: 100% accuracy, 9 iterations, 0 misclassifications

### 12.2 Deliverables
- [x] 4 Python scripts (generate, manual train, sklearn compare, pipeline)
- [x] 9 output files (4 CSV, 5 PNG)
- [x] 4 documentation files (README, PRD, prompts, requirements)
- [x] 1 configuration file (.gitignore)

### 12.3 Quality Metrics
- [x] Code comments present
- [x] All formulas implemented correctly
- [x] Visualizations are clear and labeled
- [x] README has code references
- [x] Reproducible results

---

## 13. Timeline & Milestones

| Phase | Duration | Deliverables | Status |
|-------|----------|--------------|--------|
| **Phase 1: Data Generation** | 20 min | `generate_points.py`, CSV, scatter plot | ✓ Complete |
| **Phase 2: Core Algorithm** | 35 min | Manual sigmoid/gradient/update loop | ✓ Complete |
| **Phase 3: Manual Visualizations** | 20 min | 3D surface + training metrics PNGs | ✓ Complete |
| **Phase 4: Scikit-Learn Comparison** | 15 min | Baseline model, comparison CSV/PNGs | ✓ Complete |
| **Phase 5: Integration** | 15 min | `main.py`, pipeline messaging | ✓ Complete |
| **Phase 6: Documentation** | 15 min | README, PRD, prompts | ✓ Complete |
| **Total** | **120 min** | **Complete project** | ✓ Complete |

---

## 14. Future Enhancements (Out of Current Scope)

1. **Multi-class Classification**
   - Extend to softmax regression
   - One-vs-rest strategy

2. **Regularization**
   - L1 (Lasso) penalty
   - L2 (Ridge) penalty

3. **Advanced Optimizers**
   - Momentum
   - Adam optimizer
   - Learning rate scheduling

4. **Cross-validation**
   - K-fold validation
   - Train/test/validation splits
   - Performance metrics (ROC, AUC)

5. **Interactive Visualization**
   - Plotly for 3D rotation
   - Animation of training process
   - Interactive parameter tuning

6. **Comparison Tools**
   - Benchmark against scikit-learn
   - Performance profiling
   - Ablation studies

---

## 15. Appendix

### 15.1 Glossary

| Term | Definition |
|------|------------|
| **Sigmoid** | Function that maps real numbers to (0,1) |
| **Gradient** | Vector of partial derivatives |
| **Log-likelihood** | Logarithm of probability of observed data |
| **Convergence** | When algorithm stops improving significantly |
| **Bias term** | Constant x₀=1 in linear combination |
| **Decision boundary** | Threshold where σ(x)=0.5 |

### 15.2 References

- **Formula Implementations**: `logistic_regression.py` (sigmoid/log-likelihood/gradient functions)
- **Baseline Comparison**: `logistic_regression_sklearn.py` (coefficients + visualization pipeline)
- **Development History**: `prompts.md` (15 prompts)

### 15.3 Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.1 | 2025-11-15 | Updated scope, architecture, outputs, and references to reflect comparison pipeline |
| 1.0 | 2025-11-15 | Initial PRD creation |

---

## Document Approval

| Role | Name | Status | Date |
|------|------|--------|------|
| Developer | Claude | ✓ Complete | 2025-11-15 |
| Reviewer | User | ✓ Approved | 2025-11-15 |

---

**End of Document**
