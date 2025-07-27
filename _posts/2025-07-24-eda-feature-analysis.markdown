---
layout: post
title:  "EDA: Feature Analysis"
date:   2025-07-24 11:20:00 +0800
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

Exploratory Data Analysis (EDA) is a crucial step in the machine learning project. Among various EDA aspects, **feature analysis** helps you understand the variables (features) in your dataset-how they're distributed, how they relate to each other, and which ones matter most for your downstream models. In this post, we'll cover:
1. Why feature analysis matters
2. Types of features and their characteristics
3. Univariate, bivariate, and multivariate analysis techniques
4. Dealing with missing values and outliers
5. Feature selection methods
6. Putting it all together with a Python example

## 1. Why Feature Analysis Matters

- **Data Quality Check**: Spot typos, inconsistencies, unexpected values, extreme values, and other data quality issues.
- **Insight Generation**: Discover patterns, trends, and relationships that can inform your modeling choices.
- **Model Preparation**: Identify which features need transformation, scaling, or selection.
- **Dimensionality Reduction**: Decide if some features can be dropped without losing significant information or combined into new features.

By investing time here, you ensure cleaner data, faster model convergence, and often better performance.

## 2. Know Your Feature Types

Features usually fall into two main categories: Numerical and categorical.

| Feature Type | Examples                | Common Techniques               |
|--------------|-------------------------|----------------------------------|
| Numerical    | Age, Salary, Temperature | Histograms, Boxplots, Summary Stats |
| Categorical  | Gender, Occupation, City | Bar Charts, Frequency Tables, Pivoting     |
| Datetime     | Date of Birth, Purchase Date | Time Series Plots, Feature Extraction       |
| Text         | Product Reviews, Comments | Word Counts, TF-IDF, Embeddings |

## 3. Univariate Analysis

### Numerical Features

- **Histogram**: Visualize the distribution of a numerical feature.

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```

- **Boxplot**: Identify outliers and visualize the spread.

```python
sns.boxplot(x=df['salary'])
plt.title('Salary Boxplot')
plt.xlabel('Salary')
plt.show()
```

- **Summary Statistics**: Get mean, median, standard deviation, etc.

```python
print(df['age'].describe())
```

### Categorical Features

- **Bar Chart**: Show frequency of each category.

```python
sns.countplot(x=df['gender'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.show()
```

- **Frequency Table**: Tabulate counts of each category.

```python
data['occupation'].value_counts()
```

## 4. Bivariate Analysis

### Numerical vs Numerical

- **Scatter Plot**: Visualize the relationship between two numerical features.

```python
sns.scatterplot(x=df['age'], y=df['salary'])
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()
```

- **Correlation Coefficient**: Quantify the linear relationship.

```python
correlation = df[['age', 'salary']].corr(method='pearson')
print(f'Correlation between Age and Salary: {correlation}')
```
### Numerical vs Categorical

- **Boxplot by Category**: Compare distributions across categories.

```python
sns.boxplot(x='gender', y='salary', data=df)
plt.title('Salary by Gender')
plt.xlabel('Gender')
plt.ylabel('Salary')
plt.show()
```

- **Violin Plot**: Similar to boxplot but shows the distribution shape.

```python
sns.violinplot(x='gender', y='salary', data=df)
plt.title('Salary by Gender')
plt.xlabel('Gender')
plt.ylabel('Salary')
plt.show()
```

### Categorical vs Categorical

- **Contingency Table**: Show counts of combinations of two categorical features.

```python
contingency_table = pd.crosstab(df['gender'], df['occupation'])
print(contingency_table)
```

- **Stacked Bar Chart**: Visualize the relationship between two categorical features.

```python
contingency_table.plot(kind='bar', stacked=True)
plt.title('Occupation Distribution by Gender')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.show()
```

## 5. Multivariate Analysis

When you have many features, consider:

- **Correlation Matrix/Heatmap**

```python
pearson_vars = df[['age', 'salary', 'height', 'weight']]
pearson_corr = pearson_vars.corr(method='pearson')

plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

- **Pair Plot**

```python
sns.pairplot(df[['age', 'salary', 'height', 'weight']])
plt.title('Pair Plot of Features')
plt.show()
```

- **Dimensionality Reduction (PCA)**

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[['age', 'salary', 'height', 'weight']])
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]
sns.scatterplot(x='pca1', y='pca2', data=df)
plt.title('PCA Result')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
```

## 6. Handling Missing Values & Outliers

- **Missing Values**
  - Visualize with heatmaps (sns.heatmap(df.isnull(), cbar=False))
  - Impute mean/median for numerical features, or mode for categorical features
  - Or flag with a new binary indicator column

  - **Outliers**
    - Identify via boxplots or z-score (>3 or <-3)
    - Decide: trim, winsorize, transform (e.g., log), or keep

## 7. Feature Selection Methods

Once you understand your features, you can trim or transform them:

1. **Variance Threshold**: Drop features with near-zero variance.
2. **Univariate Selection**: Use statistical tests (SelectKBest in sklearn).
3. **Recursive Feature Elimination (RFE)**: Iteratively drop leaset important features.
4. **Tree-based Importance**: Fit a random forest and rank by feature importance.
5. **Regularization**: Use Lasso or Ridge to penalize less important features.

```python
from sklearn.feature_selection import SelectKBest, f_classif
X = df.drop('target', axis=1)
y = df['target']
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)
print(selector.get_support(indices=True))  # Indices of selected features
```

## 8. Putting It All Together

```python
# 1. Load & type‐cast
df = pd.read_csv('data.csv')
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')

# 2. Univariate checks
df.describe()
df['feature'].hist()

# 3. Bivariate & multivariate
sns.boxplot(...); sns.heatmap(...)

# 4. Clean missing & outliers
df['feature'].fillna(df['feature'].median(), inplace=True)

# 5. Select features
selector = SelectKBest(f_classif, k=15)
X_sel = selector.fit_transform(X, y)
```

## Conclusion

Feature analysis in EDA is not a one-off task but an iterative process: as you build models, revisit your features: engineer new ones, drop stale ones, and always visualize to stay grounded in the data. By understanding your features deeply, you set a solid foundation for your machine learning models, leading to better performance and insights.