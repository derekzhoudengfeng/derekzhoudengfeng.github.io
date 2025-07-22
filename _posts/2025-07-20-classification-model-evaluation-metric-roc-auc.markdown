---
layout: post
title:  "Classification Model Evaluation Metric: ROC-AUC"
date:   2025-07-20 20:52:00 +0800
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

In binary classification, evaluating model performance goes beyond simply measuring accuracy at a fixed threshold. The **Receiver Operating Characteristic** (ROC) curve and its associated **Area Under the Curve** (AUC) provide a powerful, threshold-independent way to assess how well a model can discriminate between the positive and negative classes. In this post, we will delve into:
1. What is the ROC Curve and AUC?
2. How to Compute ROC-AUC in Python?
3. When to Use ROC-AUC?

## The Confusion Matrix Refresher

Before we dive into ROC, recall the binary confusion matrix:

|               | Predicted Positive | Predicted Negative |
|---------------|---------------------|---------------------|
| Actual Positive | True Positive (TP)  | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN)  |

From this, we define:
- **True Positive Rate (TPR)**: $\mathrm{TPR} = \frac{TP}{TP + FN}$ (also known as Recall or Sensitivity)
- **False Positive Rate (FPR)**: $\mathrm{FPR} = \frac{FP}{FP + TN}$ 

These quantities vary as we change the decision threshold of the model.

## What is the ROC Curve and AUC?

### ROC Curve

The **ROC Curve** is a graphical representation that illustrates the performance of a binary classifier across different thresholds. 

It plots two key metrics:
- **True Positive Rate (TPR)**:
    - On the y-axis
    - Also known as Recall or Sensitivity, it measures the proportion of actual positives correctly identified by the model.
- **False Positive Rate (FPR)**:
    - On the x-axis
    - It measures the proportion of actual negatives incorrectly identified as positives by the model.

Example ROC Curve:
![ROC Curve Example](/assets/images/roc_curve.png)

### ROC-AUC

The **ROC-AUC** (Area Under the Curve) is a single scalar value to summarize the ROC curve, quantifying the overall ability of the model to discriminate between positive and negative classes.

It ranges from 0 to 1:
- **AUC = 1**: Perfect model that distinguishes all positives from negatives.
- **AUC <= 0.5**: Model has no discriminative power, equivalent to random guessing.
- **AUC > 0.5**: Model has some discriminative power, with higher values indicating better performance.

## How to Compute ROC-AUC in Python

To compute the ROC-AUC in Python, we can use libraries like `scikit-learn`. Here's a simple example:

```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)
print(f"ROC-AUC: {roc_auc:.3f}")
```

Where:
- `y_true` is the true binary labels (0 or 1).
- `y_pred_prob` is the predicted probabilities or scores from the model.

Plotting the ROC curve can be done as follows:
```python
import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], linestyle='--', label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
```

## When to Use ROC-AUC?
- **Imbalanced Datasets**: ROC-AUC is particularly useful when dealing with imbalanced datasets, as it evaluates model performance across all classification thresholds.
- **Model Comparison**: It allows for easy comparison of different models regardless of their decision thresholds.
- **Threshold Independence**: Unlike accuracy, ROC-AUC does not depend on a specific threshold, making it a more robust metric for model evaluation.


## Summary

ROC-AUC is a powerful metric for evaluating binary classification models, especially when dealing with imbalanced datasets or varying decision thresholds. It provides a comprehensive view of model performance beyond simple accuracy, allowing for better model selection and comparison.