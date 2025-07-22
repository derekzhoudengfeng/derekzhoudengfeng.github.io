---
layout: post
title:  "Classification Model Evaluation Metric: PR-AUC"
date:   2025-07-22 15:30:00 +0800
categories: machine learning
---
<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

## Introduction

In binary classification tasks, especially with imbalanced datasets, relying solely on accuracy can be misleading. While ROC-AUC (Receiver Operating Characteristic Area Under the Curve) is widely used, it can be overly optimistic when the positive class is rare. **PR-AUC** (Precision-Recall Area Under the Curve) provides a more informative picture in such scenarios by focusing on the trade-off between precision and recall for the minority (positive) class.

## Precision and Recall Recap

Before diving into PR-AUC, let's quickly recall two core metrics it's built upon:

- **Precision**
  - The proportion of true positive predictions out of all positive predictions made by the model.
  - Formula: $\mathrm{Precision} = \frac{TP}{TP + FP}$
- **Recall (Sensitivity or True Positive Rate)**
  - The proportion of true positive predictions out of all actual positives.
  - Formula: $\mathrm{Recall} = \frac{TP}{TP + FN}$


Precision answers "Of all instances predicted positive, how many were actually positive?", while recall answers "Of all actual positives, how many did we capture?".

## Precision-Recall Curve

A Precision-Recall curve plots precision on the y-axis and recall on the x-axis for different thresholds of a binary classifier.

1. **Threshold = 1.0**: The model predicts no positives, leading to recall = 0 and precision undefined.
2. **Lowering the threshold**: The model starts predicting positives, increasing recall but often decreasing precision.
3. **Threshold = 0.0**: The model predicts all instances as positive, leading to recall = 1 and precision = $\frac{actual\ positives}{total\ instances}$.

The curve typically starts at (0, 1) and ends at (1, base rate of positives). Its shape reveals how well your model balances false positives and false negatives.

![PR Curve Example](/assets/images/pr_curve.png)

## Defining PR-AUC

PR-AUC is simply the area under the Precision-Recall curve. Formally:

$$\mathrm{PR\_AUC} = \int_0^1 \mathrm{Precision}(r) \, dr$$

- **Range**: 0 to 1
- **Interpretation**: A higher PR-AUC indicates a better model performance, especially in imbalanced datasets.
- **Baseline**: The baseline for PR-AUC is the proportion of positive instances in the dataset. A model that predicts all instances as negative will have a PR-AUC of 0.

## Why PR-AUC Matters in Imbalanced Datasets

Consider a dataset with 1% positives (e.g., fraud detection). A classifier that randomly guesses positives 1% of the time will achieve:
- ROC-AUC $\approx$ 0.5 (random chance)
- **PR-AUC** $\approx$ 0.01, which reflects the model's inability to capture the minority class effectively.

In contrast, ROC-AUC might still show a high value because it also rewards true negatives heavily. PR-AUC focuses exclusively on the positive class performance, making it a more reliable metric in such scenarios.

## Computing PR-AUC in Python

Here's a quick example using `scikit-learn` on a synthetic imbalanced dataset:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# 1. Generate imbalanced data
X, y = make_classification(n_samples=5000, n_features=20,
                           weights=[0.99], flip_y=0,
                           random_state=42)

# 2. Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 3. Predict probabilities and compute PR curve
y_scores = clf.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
avg_prec = average_precision_score(y_test, y_scores)

# 4. Plot
plt.figure(figsize=(6, 4))
plt.step(recall, precision, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'PR Curve (AP = {avg_prec:.3f})')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()
```

- Use `precision_recall_curve` to compute precision and recall at various thresholds.
- Use `average_precision_score` to compute the PR-AUC.

## Practical Tips

- **Smoothing/Rebasing**: PR curves can wiggle when thresholds change only slightly. Consider plotting the interpolated (stepwise) curve as above.
- **Baseline Awareness**: Always compare PR-AUC with positive-class rate. E.g., a PR-AUC of 0.2 on 1%-positive dataset is actually 20x better than random guessing.
- **Model Comparison**: When comparing models, use the same dataset and compare their PR-AUCs directly.
- **Threshold Selection**: The PR curve helps you pick the optimal threshold for your specific precision/recall trade-off, depending on the cost of false positives and false negatives in your application.

## Real-World Example: Fraud Detection

In credit card fraud detection:
- **Positives**: fraudulent transactions (~0.2% of total)
- **Objectives**: catch as many frauds as possible (high recall), while minimizing false alarms (reasonable precision).

A model with a PR-AUC of 0.8 dramatically outperforms random guessing (PR-AUC ~ 0.002). By inspecting the PR curve, you might choose a threshold that achieves, 75% recall with 90% precision, balancing fraud catch vs. customer inconvenience.

## Conclusion

PR-AUC is an essential metric when positives are rare and you care about correctly identifying the positive class than overall accuracy. By focusing on precision and recall, it provides clear insights into your model's ability to detect true positives without being overwhelmed by abundance of true negatives.

### Key takeaways:
- Use PR-AUC for imbalanced datasets.
- Always interpret PR-AUC relative to the positive class rate.
- Leverage the PR curve for threshold selection.
