# AI Mental Fitness Tracker

**Utilizing Machine Learning for Improved Mental Wellbeing**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Regression-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

The AI Mental Fitness Tracker is a machine learning solution designed to monitor and assess mental well-being using global health data. This project implements 12 different regression algorithms to predict mental health indicators and provide data-driven insights.

## Problem Statement

Many individuals struggle with maintaining their mental well-being due to stress, anxiety, and depression. There is a critical need for effective tools to monitor mental fitness progress, identify patterns and triggers, and provide personalized insights for proactive mental health management.

## Dataset

The project utilizes two merged datasets:
- **Mental and Substance Use Disorders Prevalence**
- **DALYs (Disability-Adjusted Life Years) - Mental Disorders**

**Data Structure:** 6,840 rows × 10 columns covering global mental health statistics across years, countries, age groups, and demographics.

## Data Visualization

### Correlation Analysis
![Correlation Heatmap](images/correlation_heatmap.png)
*Correlation heatmap showing relationships between mental health variables*

### Data Relationships
![Pairplot Analysis](images/pairplot.png)
*Pairwise relationships in the mental health dataset*

### Temporal Trends
![Year-wise Analysis](images/yearwise_bar_chart.png)
*Year-wise variations in mental fitness across different countries*

![Trend Lines](images/trend_lines.png)
*Country-specific mental health trends over time*

![Distribution](images/pie_chart.png)
*Distribution of mental health data across years*

## Machine Learning Models

### Models Implemented
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Elastic Net Regression
5. Polynomial Regression
6. Decision Tree Regression
7. Random Forest Regression
8. Support Vector Regression
9. XGBoost Regression
10. K-Nearest Neighbors Regression
11. Bayesian Regression
12. Neural Network Regression
13. Gradient Boosting Regression

### Model Performance Comparison

![Model Performance](images/model_comparison.png)
*Predicted vs Actual values comparison across all regression models*

## Results

### Performance Rankings

| Rank | Model | MSE | R² Score |
|------|-------|-----|----------|
| 1 | **Random Forest Regression** | 0.0093 | **0.9982** |
| 2 | **XGBoost Regression** | 0.0316 | 0.9938 |
| 3 | **Decision Tree Regression** | 0.0318 | 0.9938 |
| 4 | **K-Nearest Neighbors** | 0.2187 | 0.9574 |
| 5 | **Gradient Boosting** | 0.2466 | 0.9519 |
| 6 | **Polynomial Regression** | 0.3817 | 0.9256 |

**Best Model:** Random Forest Regression achieved 99.82% accuracy (R² = 0.9982)

## Installation

### Requirements
```bash
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
scikit-learn>=1.0.0
xgboost>=1.5.0
```

### Setup
```python
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load and merge datasets
df1 = pd.read_csv("mental-and-substance-use-as-share-of-disease.csv")
df2 = pd.read_csv("prevalence-by-mental-and-substance-use-disorder.csv")
data = pd.merge(df1, df2)
```

## Usage

### 1. Data Preprocessing
```python
# Handle missing values and clean data
data.dropna(inplace=True)
data.drop('Code', axis=1, inplace=True)

# Prepare features and target
X = data.drop('DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)', axis=1)
y = data['DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)']
```

### 2. Model Training
```python
# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train Random Forest (best performing model)
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
```

### 3. Evaluation
```python
# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
```

## Key Findings

- **Tree-based models** (Random Forest, XGBoost, Decision Tree) significantly outperformed linear models
- **Random Forest** achieved the highest accuracy with minimal overfitting
- **Ensemble methods** demonstrated superior generalization capabilities
- Mental health patterns show significant variation across countries and time periods

## Technology Stack

- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn, XGBoost
- **Environment:** Google Colab

## Project Structure

```
AI-Mental-Fitness-Tracker/
├── data/
│   ├── mental-and-substance-use-as-share-of-disease.csv
│   └── prevalence-by-mental-and-substance-use-disorder.csv
├── notebooks/
│   └── AI_Mental_Fitness_Tracker.ipynb
├── images/
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   └── trend_analysis.png
└── README.md
```

## Access

**Google Colab Notebook:** [View Project](https://colab.research.google.com/drive/1ENAuTwpTEJND0BilV4o9Lw6fF-_iQvwQ?usp=sharing)

## License

This project is licensed under the MIT License.

---

*This project is for educational and research purposes. Consult healthcare professionals for medical advice.*
