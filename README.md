# Kaggle-Spaceship-Titanic

This repository contains my approach to the **Kaggle Space Titanic** competition. The goal is to predict which passengers were transported to an alternate dimension using data from the spaceship's manifest.

## Exploratory Data Analysis (EDA)
The first goal was to realize an EDA (Exploratory Data Analisys) to get some valuable information about the set in general but mainly about the numerical and cathegorical features (in order to make some feature engineering if possible).

### My best insights were the following:
* **Balanced Target Variable:** The target variable (`Transported`) is almost perfectly balanced (~50/50). This is excellent news, as it means we don't need to apply synthetic oversampling techniques like SMOTE.
  ![Target Distribution](img/predicting.png)

* **Low Missing Values Rate:** The percentage of missing values (NaNs) is relatively low (<3% across all features). This allows for clean and effective imputation strategies.

* **The Power of CryoSleep:** Analyzing categorical features revealed that `CryoSleep` is a massive predictor. Passengers confined to their cryo-pods had a significantly **higher** probability of being transported.
  ![Categorical Features](img/cathegorical_plots.png)

* **The "Zero-Expense" Pattern:** For numerical features, there is a clear negative correlation between luxury spending and being transported. Passengers who spent more money on ship amenities were far less likely to be affected.
  ![Numerical Features](img/numerical_plots.png)

* **The CryoSleep & Expenditure Correlation:** At first glance, spending power seems to be the ultimate predictor. However, a deeper look reveals a logical certainty: passengers in CryoSleep *cannot* spend money. This perfect correlation means that anyone with `CryoSleep == True` has exactly $0 in expenses, explaining the spikes in our numerical distributions.
  ![Expenditure vs CryoSleep](img/spenditure_cryo.png)

## Feature Engineering & Smart Imputation

Based on the EDA insights, I applied custom feature engineering to extract hidden patterns:

1. **Smart Imputation:** Instead of blindly filling NaNs with the median/mode, I used the logical rules discovered during the EDA:
   * If a passenger's total luxury spending is > $0, missing `CryoSleep` values are imputed as `False`.
   * If `CryoSleep == True`, missing expenditure values are strictly imputed as `$0`.
2. **Cabin Splitting:** The raw `Cabin` feature (e.g., "B/0/P") was split into three distinct, highly predictive columns: `Deck`, `Num`, and `Side`.

---

## Final Pipeline
To ensure reproducibility and avoid data leakage between the train and validation sets, I implemented a `scikit-learn` Pipeline:

* **Custom Transformers:** Used `FunctionTransformer` to seamlessly integrate the Smart Imputation and Feature Engineering steps directly into the pipeline.
* **ColumnTransformer:** * **Categorical Data:** `SimpleImputer` (strategy='most_frequent') + `OneHotEncoder`.
  * **Numerical Data:** `SimpleImputer` (strategy='median') + `StandardScaler`.
* **Modeling:** Started with a baseline `RandomForestClassifier` (~79.7% accuracy) and later upgraded the core engine to an `XGBClassifier`, pushing the accuracy over the 80% mark.



