---
layout: post
title:  "ML Algorithms: Linear Regression"
date:   2025-07-28 12:30:00 +0800
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

**Linear Regression** is one of the most fundamental and widely used algorithms in machine learning. In its simplest form, it models the relationship between one (or more) independent variables (features) and a dependent variable (target) by fitting a straight line (or hyperplane) through the data points. Despite its simplicity, Linear Regression provides valuable insights, is computationally efficient, and often serves as a strong baseline for more complex models.

## 1. What is Linear Regression?

At its core, Linear Regression seeks to express the target variable $y$ as a linear combination of input features $X$:

$$\hat{y} = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$

where:
- $\hat{y}$ is the predicted output
- $w_1, w_2, ..., w_n$ are the weights assigned to each feature (coefficients)
- $x_1, x_2, ..., x_n$ are the input features
- $b$ is the bias term (Intercept)

When there is only one feature (n = 1), the model is called **Simple Linear Regression**. For multiple features (n > 1), it's called **Multiple Linear Regression**.

## 2. Mathematical Formulation

### Hypothesis Function

The model's prediction (hypothesis) is:

$$
f_{w,b}(x) = wx + b
$$

### Cost Function

To measure how well the model fits the data, we use the **Mean Squared Error (MSE)** as the cost function:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
$$

where:
- $m$ is the number of training examples
- $x^{(i)}$ and $y^{(i)}$ are the input and output for the $i^{th}$ example

## 3. Training the Model

### Gradient Descent

To minimize the cost function, we use **Gradient Descent**, which iteratively updates the parameters $w$ and $b$:

$$
w := w - \alpha \frac{\partial J(w, b)}{\partial w}
$$

$$
b := b - \alpha \frac{\partial J(w, b)}{\partial b}
$$

- $\alpha$ is the learning rate, a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.
- Repeat until convergence (i.e., until the cost function stops decreasing significantly).

### Normal Equation

For smaller datasets, you can directly solve for the optimal parameters using the **Normal Equation**:

$$
\theta = (X^T X)^{-1} X^T y
$$

where:
- $\theta$ is the vector of parameters (weights)
- $X$ is the matrix of input features
- $y$ is the vector of target values

This approach is computationally efficient for small datasets but can be infeasible for large datasets due to the matrix inversion step.

## 4. Key Assumptions

1. **Linearity**: Relationship between features and target is linear.
2. **Independence**: Observations are independent of each other.
3. **Homoscedasticity**: Constant variance of errors.
4. **Normality**: Errors are normally distributed.
5. **Minimal Multicollinearity**: Features should not be too highly correlated with each other.

Violating these assumptions can lead to biased or inefficient estimates.

## 5. Implementation in Python

Below is a step-by-step example using `scikit-learn` on a synthetic dataset:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 2. Fit the model
model = LinearRegression()
model.fit(X, y)

# 3. Make predictions
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

# 4. Evaluate
y_train_pred = model.predict(X)
mse = mean_squared_error(y, y_train_pred)
r2 = r2_score(y, y_train_pred)

print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Coefficient: {model.coef_[0][0]:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² score: {r2:.2f}")

# 5. Visualize
plt.scatter(X, y, label="Data")
plt.plot(X_new, y_pred, "r-", linewidth=2, label="Prediction")
plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.legend()
plt.show()
```

This code:
- Creates a noise linear dataset $y = 4 + 3x + \text{noise}$.
- Fits `LinearRegression()`.
- Prints the learned parameters (intercept and coefficient).
- Calculates MSE and R² score on the training data.
- Plots data points and the fitted line.

## 6. Evaluating Model Performance

- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values. Lower values indicate better fit. $MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$
- **Root Mean Squared Error (RMSE)**: Square root of MSE, providing error in the same units as the target variable. $RMSE = \sqrt{MSE}$
- **R² Score**: Represents the proportion of variance explained by the model. Values closer to 1 indicate a better fit. $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$, where $SS_{res}$ is the residual sum of squares and $SS_{tot}$ is the total sum of squares.
  - R² = 1 means perfect fit.
  - R² = 0 means the model does not explain any variability in the target variable.

## 7. Regularization Techniques

When features are many or highly correlated, regularization helps prevent overfitting:
- **Lasso Regression (L1 Regularization)**: Adds a penalty equal to the absolute value of the magnitude of coefficients. Encourages sparsity in the model.
  
  $$ J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - f_{w,b}(x^{(i)}))^2 + \lambda \sum_{j=1}^{n} |w_j| $$ 

- **Ridge Regression (L2 Regularization)**: Adds a penalty equal to the square of the magnitude of coefficients. Helps reduce model complexity.

  $$ J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - f_{w,b}(x^{(i)}))^2 + \lambda \sum_{j=1}^{n} w_j^2 $$ 

- **Elastic Net**: Combines both L1 and L2 regularization, balancing between the two.

Example in scikit-learn:

```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X, y)

# Ridge Regression
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X, y)

# Elastic Net
elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_model.fit(X, y)
```

## 8. Real-World Applications

- **House Price Prediction**: Estimating property values from size, location, and amenities.
- **Sales Forecasting**: Predicting revenue based on advertising spend across channels.
- **Medical Analytics**: Modeling patient outcomes (e.g., blood pressure) from biomakers.
- **Finance**: Risk modeling and credit scoring using borrower attributes.

## 9. Strengths & Limitations

### Strengths
- **Interpretability**: Coefficients directly show feature impact.
- **Efficiency**: Fast to train, even on large datasets.
- **Baseline Model**: Often serves as a strong baseline for regression tasks.

### Limitations
- **Linearity Assumption**: Struggles with non-linear relationships.
- **Sensitivity to Outliers**: Extreme values can skew the fitted line.
- **Multicollinearity**: High correlation among features can lead to unstable estimates.

## 10. Conclusion

Linear Regression remains a cornerstone of machine learning due to its simplicity, interpretability, and effectiveness. While more advanced models may outperform it in complex scenarios, understanding Linear Regression is essential:
1. It illustrates core concepts like cost functions and optimization.
2. It provides a benchmark for model comparison.
3. It offers insights into feature importance and relationships.

Armed with its mathematical foundation and practical implementation, you can both apply Linear Regression to real problems and build the intuition needed for more sophisticated algorithms.