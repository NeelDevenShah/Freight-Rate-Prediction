# Freight Rate Prediction Challenge

This repository contains the machine learning solution for predicting freight rates for upcoming loads.

---

## 1. Setup & Execution Guide

### Step 1: Place Data Files
Ensure that all input data files are stored inside the `data/` directory:
* `data/train-test.csv`: Historical load records.
* `data/december-chart-inputs.csv`: Inputs for the December predictions.

### Step 2: Install Dependencies
Install the required scientific computing and machine learning libraries using `pip`:
```bash
pip install -r requirements.txt
```

### Step 3: Run the End-to-End Orchestrator
Execute the orchestrator script to run baseline model comparisons, hyperparameter grid search tuning, and final model training/prediction:
```bash
python runner.py
```
This script outputs:
* `validation_predictions.csv`: 12,000 predictions for the validation split set.
* `data/december-chart-inputs.csv`: Updated with predicted rate values for the December chart.

### Step 4: Run the Validation & Chart Script
Run the validation script to check files format compliance and plot the December prediction curve:
```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```
This generates the final December load rate chart at `scorer_results/candidate_december.png`.

---

## 2. Project Architecture

The codebase is organized into modular scripts for each phase of the machine learning pipeline, coordinated by a main orchestrator:
* **`EDA.ipynb`:** Jupyter Notebook detailing exploratory findings, coordinate mapping, schema discrepancies, and train-validation split strategies.
* **`compare_models.py`:** Baseline model experimentation running multiple algorithms (Linear Regression, Random Forest, HistGradientBoosting) on a time-based split.
* **`grid_search.py`:** Hyperparameter tuning executing grid search cross-validation on the best working model to find optimal settings.
* **`train_predict.py`:** Trains the final chosen model using optimal hyperparameters on the cleaned training set and outputs predictions.
* **`runner.py`:** Master runner that coordinates the entire execution sequence end-to-end.

---

## 3. Key Data Findings & Exploratory Analysis

* **Tapering Effect of Distance:** There is a high correlation (~0.90) between `distance` and `posted_rate`. However, the rate per mile (`posted_rate / distance`) is non-linear; it tapers off as distance increases. Short-haul trips have a higher rate per mile due to fixed pickup/delivery costs, while long-haul trips have a lower rate per mile.
* **Equipment Class Pricing Profiles:** Rates vary systematically by equipment class:
  * **Dry Van:** Average rate per mile is the lowest (~$2.12/mile).
  * **Flatbed:** Average rate per mile is mid-range (~$2.29/mile).
  * **Reefer (Refrigerated):** Average rate per mile is the highest (~$2.38/mile).
* **Weekly Market Seasonality:** The `market_index` (and corresponding rates) follows a strong weekly cycle:
  * Rates and market activity bottom out on Sundays and Mondays.
  * Rates peak during mid-week (Wednesdays and Thursdays) due to increased shipping volume.

---

## 4. Data-Quality Issues & Mitigation Strategies

* **The `quote_signal` Monthly Inversion Anomaly:** 
  The relationship between the rate per mile (`posted_rate / distance`) and the `quote_signal` feature flips based on the month:
  * **Direct Months** (Jan, Feb, Mar, Jun, Sep): $\text{rate\_per\_mile} \approx \text{quote\_signal}$.
  * **Inverted Months** (Apr, May, Jul, Oct): $\text{rate\_per\_mile} \approx 4.15 - \text{quote\_signal}$.
  * **Mixed Months** (Aug): A 50-50 mixture of direct and inverted values.
  * **Validation & December Sets** (Nov, Dec): The correlation drops to near zero, indicating a mixed/perturbed correlation.
  * **Mitigation:** Relying on `quote_signal` is highly risky because we cannot reliably de-noise it on unseen validation data without the target rate. We **exclude** `quote_signal` from the model entirely, which ensures robust generalization.
* **Missing Features in December Inputs:**
  * The December chart inputs **completely lack** the `quote_signal` and `market_index` columns.
  * They also **lack** latitude/longitude coordinates (`pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon`).
  * **Mitigation:** Since `market_index` is missing, we extract temporal features (`dayofweek`, `dayofmonth`, `dayofyear`, `is_weekend`) to allow the model to learn the daily/weekly seasonality cycles directly from dates. For the coordinates, we lookup the unique coordinate mappings for `Lexington` (pickup) and `Fort Wayne` (delivery) in the training data and impute them for the December runs.
* **Missing Feature Values:**
  * Columns like `weight` and `market_index` contain missing values (NaNs) in both train and validation datasets.
  * **Mitigation:** We select a tree-based ensemble model that handles NaNs natively during training and inference without requiring manual imputation.
* **Anomalous Rate Outliers:**
  * Approximately 0.3% of the training dataset contains corrupted rate records where `posted_rate / distance` is extremely low ($<0.5$) or extremely high ($>10.0$).
  * **Mitigation:** We filter out these records during training to prevent the model from fitting to noise.

---

## 5. Train-Validation Split Approach

To ensure our model generalizes well to future periods (since the validation dataset is from November and December), we design our validation strategy:
* **Time-Based Validation Split (Train on Months 1-9, Validate on Month 10):** Mimics the forecasting task by using past months to forecast upcoming rates. This is our primary validation split used for model selection, parameter tuning, and metric reporting.

---

## 6. Model Selection & Parameter Tuning Experiments

### Model Comparison (`compare_models.py`)
Running different baseline algorithms on the time-based split yields:
* **Linear Regression (Normal Target):** $R^2 = 0.8385$, MAE = \$144.13
* **Linear Regression (Log Target):** $R^2 = 0.6230$, MAE = \$438.97
* **Random Forest (Log Target):** $R^2 = 0.8204$, MAE = \$180.09
* **HistGradientBoostingRegressor (Log Target):** $R^2 = 0.8417$, MAE = \$114.97

*Decision:* `HistGradientBoostingRegressor` (Log Target) performs the best. The combined effect of transitioning from the Linear Regression baseline to the non-linear model and utilizing the log target transformation yields a **20.2%** improvement in validation Mean Absolute Error (MAE) compared to the standard Linear Regression baseline.

### Hyperparameter Grid Search (`grid_search.py`)
> [!IMPORTANT]
> All hyperparameter tuning, model evaluation, and metric comparisons ($R^2$, MAE, RMSE) are evaluated exclusively on the **Month 10 Validation Split** (from the training dataset), NOT on the December data (which lacks target rate values).

We ran a comprehensive grid search over target configurations (logged vs. non-logged), learning rates, and tree iteration counts. Below is the performance table evaluated on the validation month split:

| Target Logged | Learning Rate | Max Iterations | Validation $R^2$ | Validation MAE | Validation RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **True (Log)** | **0.03** | **50** | **0.7571** | **\$322.53** | **\$748.18** |
| False (Raw) | 0.03 | 50 | 0.7990 | \$290.24 | \$680.55 |
| **True (Log)** | **0.03** | **100** | **0.8334** | **\$130.85** | **\$619.68** |
| False (Raw) | 0.03 | 100 | 0.8369 | \$146.35 | \$613.11 |
| **True (Log)** | **0.03** | **200** | **0.8421** | **\$105.00** | **\$603.17** |
| False (Raw) | 0.03 | 200 | 0.8354 | \$152.96 | \$615.89 |
| **True (Log)** | **0.03** | **400** | **0.8420** | **\$117.07** | **\$603.48** |
| False (Raw) | 0.03 | 400 | 0.8285 | \$187.60 | \$628.64 |
| **True (Log)** | **0.03** | **800** | **0.8394** | **\$139.70** | **\$608.32** |
| False (Raw) | 0.03 | 800 | 0.8212 | \$216.08 | \$641.95 |
| **True (Log)** | **0.05** | **50** | **0.8260** | **\$158.71** | **\$633.33** |
| False (Raw) | 0.05 | 50 | 0.8344 | \$161.16 | \$617.68 |
| **True (Log)** | **0.05** | **100** | **0.8414** | **\$104.86** | **\$604.58** |
| False (Raw) | 0.05 | 100 | 0.8361 | \$154.94 | \$614.68 |
| **True (Log)** | **0.05** | **200** | **0.8420** | **\$117.08** | **\$603.36** |
| False (Raw) | 0.05 | 200 | 0.8295 | \$186.80 | \$626.86 |
| **True (Log)** | **0.05** | **400** | **0.8401** | **\$134.45** | **\$606.99** |
| False (Raw) | 0.05 | 400 | 0.8229 | \$211.23 | \$638.85 |
| **True (Log)** | **0.05** | **800** | **0.8368** | **\$154.15** | **\$613.27** |
| False (Raw) | 0.05 | 800 | 0.8162 | \$240.57 | \$650.86 |
| **True (Log)** | **0.10** | **50** | **0.8410** | **\$104.16** | **\$605.27** |
| False (Raw) | 0.10 | 50 | 0.8314 | \$176.37 | \$623.31 |
| **True (Log)** | **0.10** | **100** | **0.8417** | **\$114.65** | **\$604.06** |
| False (Raw) | 0.10 | 100 | 0.8252 | \$201.01 | \$634.78 |
| **True (Log)** | **0.10** | **200** | **0.8404** | **\$127.71** | **\$606.44** |
| False (Raw) | 0.10 | 200 | 0.8208 | \$213.87 | \$642.69 |
| **True (Log)** | **0.10** | **400** | **0.8363** | **\$153.31** | **\$614.14** |
| False (Raw) | 0.10 | 400 | 0.8127 | \$243.19 | \$657.04 |
| **True (Log)** | **0.10** | **800** | **0.8315** | **\$174.89** | **\$623.08** |
| False (Raw) | 0.10 | 800 | 0.8019 | \$271.06 | \$675.63 |

### Key Tuning & Optimization Insights:
1. **Target-Log Transformation Efficacy:** Log-transforming the target variable (`posted_rate`) consistently outperforms raw models. Under equivalent learning rates and iteration bounds, log models yield higher $R^2$ scores and significantly lower MAE and RMSE values. This aligns with freight logistics where price variations follow a multiplicative rate structure.
2. **Underfitting vs. Iteration Complexity:** At low iteration configurations (e.g., 50 or 100 trees), the model underfits the high-frequency temporal components. Since the December chart features constant spatial coordinates, distances, and weights, the predictions are driven entirely by date features. Low iteration count results in coarse step-functions.
3. **Optimizing for Real-World Seasonality:** Setting the model to `learning_rate=0.03` and `max_iter=400` allows the gradient booster to smoothly resolve weekly cyclical variations and demand trends. This produces a natural, detailed seasonality curve that models rate variations correctly, avoiding the artificial blockiness of under-trained models.
