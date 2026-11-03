[README_Hospital_LOS_Prediction.md](https://github.com/user-attachments/files/34275499/README_Hospital_LOS_Prediction.md)
# Hospital Length of Stay (LOS) Prediction

## Project Overview

This project develops and evaluates machine learning regression models to predict the **length of hospital stay (LOS), measured in days**, using patient and hospital-admission information. The analysis is presented in the Jupyter notebook `Hospital_LOS_Prediction.ipynb`.

The project uses a hospital-management scenario involving HealthPlus Hospital. Its practical motivation is to support better planning of beds, staffing, wards, and other hospital resources by estimating how long admitted patients may remain in the hospital.

The notebook combines exploratory data analysis (EDA), categorical-variable encoding, regression diagnostics, model comparison, hyperparameter tuning, and interpretation of model results.

## Project Objectives

1. Explore the patient and hospital-admission dataset.
2. Identify patient, admission, and hospital characteristics associated with length of stay.
3. Prepare the data for regression modelling, including encoding categorical variables.
4. Build and evaluate multiple regression algorithms.
5. Compare linear, tree-based, ensemble, and boosting models.
6. Examine feature importance and derive operational insights relevant to hospital resource planning.

## Dataset

Too big to be uploaed

The dataset contains **500,000 records and 15 columns**. The recorded notebook output reports no missing values and no duplicate rows.

The target variable is:

- `Stay (in days)` — the patient's length of stay in the hospital, measured in days.

The dataset includes the following fields:

| Variable | Description |
|---|---|
| `patientid` | Patient identifier |
| `Age` | Patient age group |
| `gender` | Patient gender |
| `Type of Admission` | Trauma, emergency, or urgent |
| `Severity of Illness` | Minor, moderate, or extreme |
| `health_conditions` | Recorded pre-existing health condition(s) |
| `Visitors with Patient` | Number of visitors accompanying the patient |
| `Insurance` | Whether the patient has insurance |
| `Admission_Deposit` | Deposit paid at admission |
| `Available Extra Rooms in Hospital` | Number of extra rooms available |
| `Department` | Hospital department |
| `Ward_Facility_Code` | Ward facility identifier |
| `doctor_name` | Treating doctor's name |
| `staff_available` | Number of staff available |
| `Stay (in days)` | Length of hospital stay; prediction target |

The notebook removes `patientid` from the predictors because it is an identifier rather than a substantive explanatory feature. Categorical variables are converted into dummy variables using `pandas.get_dummies()` with `drop_first=True`.

**Dataset availability:** `healthcare_data.csv` is an input required to run the notebook. It is not included merely by uploading the notebook. Add the CSV to the repository only if you have the right to redistribute it; otherwise, provide instructions for obtaining it from its original source.

## Methodology

### 1. Data loading and inspection

The analysis loads the CSV file with pandas and examines its dimensions, column types, sample records, descriptive statistics, category distributions, missing values, and duplicate records.

### 2. Exploratory Data Analysis

Univariate and bivariate analyses are used to explore the distribution of length of stay and other variables. The notebook examines patient age groups, gender, admission type, illness severity, department, ward, visitors, admission deposits, and hospital resources.

### 3. Data preparation

- Categorical variables are transformed into dummy variables.
- The patient identifier (`patientid`) is removed from the modelling data.
- The target (`Stay (in days)`) is separated from the predictor variables.
- The data is divided into training and testing subsets using an 80:20 split and `random_state=1`.

The recorded split contains **400,000 training observations and 100,000 testing observations**, with 41 predictor columns.

### 4. Regression diagnostics

The notebook builds an Ordinary Least Squares (OLS) regression model and examines model assumptions and diagnostics. These include multicollinearity using Variance Inflation Factors (VIF), residual behaviour and normality, linearity, and homoscedasticity using the Goldfeld–Quandt test.

It also applies 10-fold cross-validation to assess the linear regression model.

### 5. Model development and comparison

The notebook evaluates the following regression approaches:

- Ordinary Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Bagging Regressor
- Random Forest Regressor
- Tuned Decision Tree Regressor
- Tuned Bagging Regressor
- Tuned Random Forest Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Grid search with cross-validation is used to tune selected decision tree, bagging, and random forest models.

### 6. Evaluation metrics

Models are evaluated using:

- **RMSE (Root Mean Squared Error):** measures prediction error in days and penalises larger errors more heavily.
- **MAE (Mean Absolute Error):** measures the average absolute prediction error in days.
- **R-squared:** measures the proportion of target variation explained by the model.
- **Adjusted R-squared:** adjusts R-squared for the number of predictors.
- **MAPE (Mean Absolute Percentage Error):** expresses absolute prediction error as a percentage of actual LOS.

## Results

The following values are taken from the notebook's recorded test-set model comparison. Values are rounded to three decimal places.

| Model | RMSE | MAE (days) | R-squared | MAPE (%) |
|---|---:|---:|---:|---:|
| Linear Regression | 3.144 | 2.156 | 0.843 | 19.677 |
| Ridge Regression | 3.144 | 2.156 | 0.843 | 19.678 |
| Decision Tree Regressor | 1.815 | 1.128 | 0.948 | 9.341 |
| Random Forest Regressor | **1.304** | **0.865** | **0.973** | **7.315** |
| Bagging Regressor | 1.369 | 0.905 | 0.970 | 7.649 |
| AdaBoost Regressor | 2.375 | 1.587 | 0.910 | 13.624 |
| Gradient Boosting Regressor | 1.793 | 1.213 | 0.949 | 10.247 |
| XGBoost Regressor | 1.511 | 1.033 | 0.964 | 8.867 |

Based on the notebook's test-set results, the **Random Forest Regressor with 100 trees and `random_state=1`** is selected as the final model. It achieves:

- Test RMSE: approximately **1.304 days**
- Test MAE: approximately **0.865 days**
- Test R-squared: approximately **0.973**
- Test adjusted R-squared: approximately **0.973**
- Test MAPE: approximately **7.315%**

The model therefore explains approximately 97.3% of the variation in LOS in the recorded test set. Its MAE indicates an average absolute prediction error of about 0.87 days on that test set.

The tuned Random Forest model in the notebook performs less well than the default Random Forest configuration on the reported test metrics. This illustrates why model selection should be based on held-out evaluation rather than assuming that hyperparameter tuning will always improve performance.

## Feature Importance and Operational Insights

The notebook uses feature-importance plots from the tuned Decision Tree and Random Forest models to examine which predictors contribute to model decisions. Its discussion highlights department, age group, gender, and doctor-related variables, as well as patient visitors and admission deposit, as potentially useful operational factors.

The EDA and notebook discussion also identify areas for hospital planning, including:

- The large share of admissions associated with the gynecology department.
- Differences in length of stay across wards and age groups.
- The possible resource implications of visitor numbers.
- The need to consider staffing, available rooms, and emergency admissions when planning capacity.

These are exploratory and model-based findings, not evidence that changing any one factor will necessarily reduce LOS. Feature importance indicates predictive contribution in a fitted model; it does not establish causality.

## How to Run the Project

### Requirements

- Python 3
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- statsmodels
- XGBoost

Install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost jupyter
```

### Run locally

1. Clone or download this repository.
2. Place `healthcare_data.csv` in the same directory as the notebook.
3. Install the dependencies.
4. Start Jupyter Notebook:

## Limitations and Recommended Further Work

The notebook is an exploratory machine learning project. Before treating its results as evidence of real-world predictive performance, consider the following:

1. **Repeated patient records:** The dataset contains repeated admissions for some patient IDs. Although `patientid` is removed as a predictor, the notebook uses a random row-level train/test split. Records belonging to the same patient may therefore appear in both sets. A patient-grouped split would provide a stronger test of generalisation to previously unseen patients.
2. **Prediction-time availability:** Confirm that every predictor is genuinely available at the intended time of prediction. Variables recorded after admission or during the hospital stay could make the model unsuitable for an admission-time forecast.
3. **Encoding before splitting:** Dummy encoding is performed before the train/test split. For a stricter evaluation pipeline, fit preprocessing only on training data, preferably through a scikit-learn `Pipeline` and `ColumnTransformer`.
4. **Tuning and validation:** Keep the test set untouched during model selection. Use cross-validation within the training data for tuning, then evaluate the selected model once on the held-out test set.
5. **Feature importance:** Tree-based feature importance can be biased toward certain feature types and should not be interpreted as causal impact. Permutation importance or SHAP analysis could provide additional interpretation.
6. **Metric interpretation:** The notebook's MAPE calculation divides by actual LOS. It is unsuitable when actual target values are zero; verify the target values and metric assumptions before reuse on another dataset.
7. **Clinical and operational validation:** Validate the model prospectively and with relevant hospital stakeholders before using predictions to influence staffing, bed allocation, discharge planning, or patient care.

The model is a decision-support research exercise, not a clinical diagnostic system. Predictions should not replace professional clinical judgement.


**Opoku Asante Emmanuel**, Data Scientist and Financial Economist

Email: vkpoasante@gmail.com

## License
**MIT**
