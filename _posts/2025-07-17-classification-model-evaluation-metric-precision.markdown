---
layout: post
title:  "Classification Model Evaluation Metric: Precision"
date:   2025-07-17 14:40:00 +0800
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

In many real-world classification tasks, especially those involving imbalanced datasets, **accuracy** alone can be misleading. A model might achieve high accuracy by simply predicting the majority class most of the time. Therefore, it is essential to consider other metrics such as **precision**. Precision is particularly important when the **cost of false positives is high**, such as in email spam detection.

In this post, we will explore:
1. What is Precision?
2. When to Use Precision?
3. Implementing Precision in Python

## What is Precision?

> Precision is the proportion of correct positive predictions out of all positive predictions.

The formula for calculating precision is:

$$
\mathrm{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} = \frac{TP}{TP + FP}
$$

where:
- **TP (True Positives)**: model correctly predicts the positive class
- **FP (False Positives)**: model incorrectly predicts the positive class

A high precision means that when your model says "positive", it is usually correct. This is crucial in scenarios where false positives can lead to significant costs or consequences.

## When to Use Precision?

You should consider using precision as your primary metric when the cost of false alarm is high. 

For example, in email spam detection, if a valid email is incorrectly classified as spam (false positive), it could lead to missing important email. In such case, you would prefer a model that minimizes false positives, even if it means sacrificing some true positives.

## Implementing Precision in Python

You can easily calculate precision using **scikit-learn**:

```python
from sklearn.metrics import precision_score

# True labels
y_true = [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]

# Predicted labels
y_pred = [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]

# Calculate precision
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.2f}")
# Output: Precision: 0.83
```

## Summary

Precision is a critical metric in classification tasks, especially when the cost of false positives is high. It provides a more nuanced view of model performance compared to accuracy, particularly in imbalanced datasets. By focusing on precision, you can ensure that your model's positive predictions are reliable and actionable.