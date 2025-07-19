---
layout: post
title:  "Classification Model Evaluation Metric: F1-Score"
date:   2025-07-19 20:50:00 +0800
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

Machine learning practitioners often face classification problems where simply measuring accuracy is not enough-especially when classes are imbalanced or the costs of different errors vary. The **F1-Score** combines two key metrics: **precision** and **recall**, into a single score that balances both concerns. In this post, we will explore:
1. What is Precision and Recall?
2. Why Accuracy Can Be Misleading?
3. What is the F1-Score?
4. Practical Example
5. Implementation in Python
6. When to Use the F1-Score?

## What is Precision and Recall?

- **Precision**
    - Precision is the proportion of correct positive predictions out of all positive predictions made by the model.
    - The formula: $\mathrm{Precision} = \frac{TP}{TP + FP}$
    - A high precision means few false alarms (FP).
- **Recall** 
    - Recall is the proportion of correct positive predictions out of all actual positives.
    - The formula: $\mathrm{Recall} = \frac{TP}{TP + FN}$
    - A high recall means few missed positives (FN).

> Why both?
> - Precision focuses on the correctness of positive predictions.
> - Recall focuses on capturing as many actual positives as possible.

## Why Accuracy Can Be Misleading

Consider a disease detection where only 1% of the patients have the disease. A model that predicts everyone as healthy will have 99% accuracy-but it never detects the disease! In such imbalanced scenarios, accuracy can be misleading, and precision and recall (and thus F1-Score) provide a far clearer picture of model performance.

## What is the F1-Score?

> F1-Score is the harmonic mean of precision and recall, providing a single metric that balances both.

The formula for F1-Score is:

$$
\mathrm{F1} = 2 \cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}} = 2 \cdot \frac{TP}{2TP + FP + FN}
$$

- Harmonic mean penalizes extreme values: if either precision or recall is low, the F1-Score will also be low.
- Range from 0 to 1, where 1 is perfect model and 0 is useless model.

## Practical Exmaple

Suppose on a test set, we have the following results:
- TP = 40
- FP = 10
- FN = 50
- TN = 900

Then:
1. Precision = $\frac{40}{40 + 10} = 0.8$
2. Recall = $\frac{40}{40 + 50} = 0.44$
3. F1-Score = $2 \cdot \frac{0.8 \cdot 0.44}{0.8 + 0.44} \approx 0.57$

Despite high precision (few false positives), recall is low (many false negatives), and the F1-Score reflects this by being moderate.

## Implementation in Python

You can calculate the F1-Score using **scikit-learn**:

```python
from sklearn.metrics import f1_score

# True labels
y_true = [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]

# Predicted labels
y_pred = [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]

# Calculate F1-Score
f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1:.2f}")
# Output: F1-Score: 0.83
```

## When to Use the F1-Score?

- Dataset is imbalanced (e.g., fraud detection, disease diagnosis).
- False positives and false negatives have similar costs.

## Summary

The F1-Score provides a balanced view of model performance by combining precision and recall into a single metric. It is particularly useful in scenarios where the classes are imbalanced and the cost of false positives and false negatives are similar. By focusing on both aspects, the F1-Score helps ensure that models are not just accurate, but also effective in capturing the positive class.