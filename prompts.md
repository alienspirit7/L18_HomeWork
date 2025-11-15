# Prompts and Actions Summary

## 1. Initial Script Creation
**Prompt:** *"Create a script that creates a set of random points (x1,x2). This data set should create 2 groups of points that is easy to split into 2 groups. The result of the script is dataframe x1, x2, R, where R is 1 or 0, depending on to which group the point belongs to. x1 and x2 should be values between 0 and 1."*

**Actions:** Created `generate_points.py` with two Gaussian clusters, DataFrame output (x1, x2, R), CSV export, and visualization.

---

## 2. Increase Sample Size
**Prompt:** *"Make sample size to be 5000 points"*

**Actions:** Updated `n_points_per_group` to 5000 (10,000 total points).

---

## 3. Create Requirements File
**Prompt:** *"create requirements.txt for this script"*

**Actions:** Created `requirements.txt` with numpy, pandas, matplotlib.

---

## 4. Increase Group Separation
**Prompt:** *"Make groups be further from each other"*

**Actions:** Moved clusters to (0.2, 0.2) and (0.8, 0.8), reduced std to 0.08.

---

## 5. Adjust Sample Size
**Prompt:** *"Make sample size be 2000 each group"*

**Actions:** Changed `n_points_per_group` to 2000 (4,000 total points).

---

## 6. Organize Output
**Prompt:** *"Make all the results to be saved into the folder, results_output"*

**Actions:** Added output directory creation, moved CSV and plot to `results_output/` folder.

---

## 7. Version Control Setup
**Prompt:** *"Add gitignore"*

**Actions:** Created `.gitignore` for Python cache, virtual environments, IDE files, and OS files.

---

## 8. Custom Environment Exclusion
**Prompt:** *"add to gitignore myenv"*

**Actions:** Added `myenv/` to `.gitignore`.

---

## 9. Track Results in Git
**Prompt:** *"allow to git results output"*

**Actions:** Removed `results_output/` from `.gitignore` to track output files.

---

## 10. Rename Documentation
**Prompt:** *"It should not be README file, it should be prompts.md file"*

**Actions:** Renamed README.md to prompts.md.

---

## 11. Simplify Documentation
**Prompt:** *"In prompts.md file only keep prompts used and short summary of the actions taken"*

**Actions:** Simplified prompts.md to show only prompts and brief action summaries.

---

## 12. Create Logistic Regression Script
**Prompt:** *"Now use the formulas in Screenshot files in folder (3 files, 3 formulas). Use these formulas and file created by script generate_points.py (x0 which is bias is always 1) and create the process of gradient decent where In each iteration: we compute z (a weighted sum of the features and coefficients); Apply the sigmoid to obtain the probability 𝑝; Take the gradient to derive the loss function's derivative; Update the coefficients according to the learning step. This is how the algorithm adjusts the coefficient values that minimize the logistic-loss function for the given data. The script should save all the iterations tries and then provide the following: 1) Plot of the data set where points have color of initial groups (0-blue or 1-red) and form of the groups defined by the sigmoid (if 0-round, if 1-triangle); 2) Graph of how error and likelyhood function changed over the iterations; 3) Dataset table (x1, x2, R (from original dataset), sigma(x1,x2), error with Average Error in the bottom)."*

**Actions:** Created `logistic_regression.py` implementing gradient descent with sigmoid, gradient, and coefficient update formulas; saves iteration history, results table, classification plot, and training metrics graphs.

---

## 13. Add Early Stopping
**Prompt:** *"Stop iterations when error is small enough"*

**Actions:** Added convergence check with `error_threshold=0.01`; stops training when average error drops below threshold; increased max iterations to 10,000.

---

## 14. Convert to 3D Visualization
**Prompt:** *"The plot of 2 groups should be in 3D and split should be sigmoid function"*

**Actions:** Modified visualization to 3D plot with sigmoid surface; z-axis shows probability σ(x); plotted sigmoid surface and decision boundary at z=0.5.

---

## 15. Combine Training Metrics
**Prompt:** *"Combine error function and likelihood function into the same graph"*

**Actions:** Merged two subplots into single graph with dual y-axes; left axis for average error, right axis for log-likelihood.
