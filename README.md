# Loan Default Prediction with a Deep Neural Network (from scratch in NumPy)

A deep learning mini project in the banking/credit-risk domain: predict whether a borrower will default on a loan, using a multi-layer neural network whose forward pass, backpropagation and Adam optimiser are all implemented by hand in NumPy, then benchmarked against Logistic Regression.

## What's inside

| File | Purpose |
|---|---|
| `loan_default_dnn.py` | Full pipeline: data generation, preprocessing, DNN, gradient check, evaluation, plots |
| `loan_data.csv` | Synthetic dataset (20,000 loans, ~15% default rate) |
| `metrics.json` | All final metrics and feature importances |
| `results.png` | Training curve, ROC, confusion matrix, feature importance |

Run it: `pip install numpy pandas scikit-learn matplotlib` then `python loan_default_dnn.py` (about 15 seconds on CPU).

## Approach

- **Data:** synthetic borrower records (age, income, CIBIL-style credit score 300-900, EMI-to-income ratio, past delinquencies, employment years, credit lines, loan term/amount, home ownership, loan purpose). Default risk is generated from non-linear rules plus noise. No real customer data.
- **Preprocessing:** log transform of income and loan amount, one-hot encoding, standardisation fitted on the training set only (no leakage). Stratified 70/15/15 train/validation/test split.
- **Model:** MLP `17 -> 64 -> 32 -> 16 -> 1` with ReLU, He initialisation, inverted dropout (0.2), L2 regularisation, sigmoid output.
- **Training:** mini-batch Adam, class-weighted binary cross-entropy for the 15/85 imbalance, early stopping on validation loss.
- **Verification:** numerical gradient check confirms backprop is correct (max relative error 9e-9).
- **Decision threshold:** chosen on the validation set to maximise F1, then applied unchanged to the test set.

## Results (held-out test set, 3,000 loans)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Deep NN | 0.904 | 0.721 | 0.619 | 0.682 | 0.649 |
| Logistic Regression | 0.906 | 0.726 | 0.604 | 0.702 | 0.649 |

DNN confusion matrix: 2,367 repaid correctly, 187 false alarms, 142 missed defaults, 304 defaults caught.

**Most important features** (permutation importance, drop in test AUC): EMI-to-income ratio (0.205), credit score (0.129); everything else is below 0.005.

## Honest takeaways

- The DNN matches but does not beat logistic regression here. The synthetic data is dominated by two roughly monotonic signals (EMI burden and credit score), which a linear model captures well. Deep networks tend to pay off with more features, larger data and stronger interactions, so on real bank data the gap could differ.
- The model flags defaults well (about 68% recall) at the cost of some false alarms. In a real lending setting the threshold would be set from the relative cost of a missed default versus a rejected good customer.
- Results come from synthetic data, so they show the method works, not how it would perform on a real loan book.

## Ideas to extend

- Replace the synthetic data with a public dataset (e.g. Lending Club or German Credit) and re-run.
- Port the model to PyTorch or TensorFlow and compare.
- Add SHAP explanations, probability calibration, or a cost-based threshold.
- Build a Power BI dashboard on `loan_data.csv` plus model scores.
