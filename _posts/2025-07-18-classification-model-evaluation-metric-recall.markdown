---
layout: post
title:  "Classification Model Evaluation Metric: Recall"
date:   2025-07-18 13:00:00 +0800
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

In the world of machine learning, evaluating a classification model's performance involves more than just measuring accuracy. Different applications care differently about the types of mistakes a model makes. **Recall**, also known as **sensitivity** or **true positive rate**, is one such metric that shines in scenarios where missing a positive instance is costly.

In this post, we will explore:
1. What is Recall?
2. When to Use Recall?
3. How to Implement Recall in Python?

## What is Recall?

> Recall is the proportion of correct positive predictions out of all actual positives.

The formula for calculating recall is:

$$
\mathrm{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} = \frac{TP}{TP + FN}
$$

where:
- **TP (True Positives)**: model correctly predicts the positive class
- **FN (False Negatives)**: model incorrectly predicts the negative class

A high recall (close to 1) means that the model missed very few positive instances.

## When to Use Recall?

Imagine a medical test for a serious disease. You'd rather **flag as many sick patients as possible** (even if some healthy patients are flagged incorrectly) than miss someone who actually has the disease. In such cases, recall is more important than precision.

In general, you should consider using recall as your primary metric when:
- You want to ensure that you capture as many positive instances as possible (Cost of missing a positive instance is high).
- You are dealing with imbalanced datasets where the positive class is rare.

## How to Implement Recall in Python

You can easily calculate recall using **scikit-learn**:

```python
from sklearn.metrics import recall_score

# True labels
y_true = [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]

# Predicted labels
y_pred = [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]

# Calculate recall
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.2f}")
# Output: Recall: 0.83
```

## Summary

Recall is a fundamental metric in classification tasks, especially when the cost of missing positive instances is high. It helps ensure that your model captures as many positive cases as possible, making it a crucial metric in fields like healthcare, fraud detection, and more.