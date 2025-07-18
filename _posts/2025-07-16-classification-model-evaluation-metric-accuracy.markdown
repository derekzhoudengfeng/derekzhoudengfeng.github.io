---
layout: post
title:  "Classification Model Evaluation Metric: Accuracy"
date:   2025-07-16 11:20:00 +0800
categories: evaluation metrics
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

In supervised machine learning, evaluating how well a classification model performs is crucial. One of the most commonly used metrics is **accuracy**, which measures the proportion of correct predictions out of all predictions made by the model. It is simplest and most intuitive metric, but it has its limitations, especially in cases of imbalanced datasets.

## What is Accuracy?

> Accuracy is the proportion of correct predictions out of total predictions.

The formula for calculating accuracy is:

$$
\mathrm{Accuracy}
\;=\;
\frac{\text{Correct Predictions}}
     {\text{Total Predictions}}
\;=\;
\frac{TP + TN}{TP + TN + FP + FN}
$$

where:
- **TP (True Positives)**: model correctly predicts the positive class
- **TN (True Negatives)**: model correctly predicts the negative class
- **FP (False Positives)**: model incorrectly predicts the positive class
- **FN (False Negatives)**: model incorrectly predicts the negative class

## Implementing Accuracy in Python

A quick way to calculate accuracy using **scikit-learn**:

```python
from sklearn.metrics import accuracy_score

# True labels
y_true = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]

# Predicted labels
y_pred = [1, 0, 1, 0, 0, 0, 1, 0, 1, 1]

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# Output: Accuracy: 0.80
```

## When to Use Accuracy

1. **Balanced Datasets**: When the number of samples in each class is roughly equal, accuracy is a reliable metric.
2. **Symmetric Costs**: If the false positives and false negatives have similar real-world costs, accuracy's equal weighting of all errors is appropriate.
3. **Quick Baseline**: As one of the simplest metrics, accuracy is a convinient first check before diving into more nuanced metrics.

## Pitfalls of Accuracy

While accuracy is easy to understand and compute, it has significant limitations:

- **Imbalanced Datasets**: 
    - In cases where one class significantly outnumbers another, accuracy can be misleading. 
    - For example, a fraud detection model where 1% of transactions are fraudulent. A naive model that predicts all transactions as non-fraudulent would achieve 99% accuracy, but it is useless in practice.
- **Unequal Error Costs**:
    - If the cost of false positives is much higher than false negatives (or vice versa), accuracy does not reflect the model's effectiveness in minimizing the more costly errors.
    - For example, in medical diagnosis, missing a disease (false negative) may be far worse than a false alarm (false positive). Accuracy treats both errors equally, which may not be appropriate.

## Alternatives and Complements

When accuracy alone is not enough, consider using these metrics:
- **Precision**: $\frac{TP}{TP + FP}$ - proportion of correct positive predictions out of all predicted positives.
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ - proportion of correct positive predictions out of all actual positives.
- **F1 Score**: Harmonic mean of precision and recall, balancing both.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve, evaluates the trade-off between true positive rate and false positive rate across thresholds.

## Conclusion

Accuracy is a fundamental metric in classification model evaluation, providing a quick overview of model performance. However, it is essential to understand its limitations, especially in the context of imbalanced datasets or when the costs of different types of errors are not equal. Always consider complementing accuracy with other metrics to get a more comprehensive view of your model's performance.