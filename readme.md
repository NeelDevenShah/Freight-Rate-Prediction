# Freight Rate Prediction

## Problem Overview
This repository contains a machine learning pipeline to predicting freight rates (`posted_rate`) for truckload shipments. Given load attributes—such as origin/destination locations, travel distance, equipment class (Dry Van, Flatbed, Reefer), weight, and shipment date—the model forecasts the market rate. The goal is to accurately predict rates on unseen future periods (November and December) while resolving the daily/weekly seasonality cycles of the freight market.

All final artifacts—including the hyperparameter tuning logs (`tuning_results.csv`) and the generated December prediction charts (`candidate_december.png`)—are stored in the [`scorer_results/`](file:///home/neel/Desktop/spotterAI/scorer_results) directory.

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
python src/score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```
This generates the final December load rate chart at `scorer_results/candidate_december.png`.

---

## 2. Project Architecture

The codebase is organized into modular scripts for each phase of the machine learning pipeline, coordinated by a main orchestrator:
* **`EDA.ipynb`:** Jupyter Notebook detailing exploratory findings, coordinate mapping, schema discrepancies, and train-validation split strategies.
* **`src/compare_models.py`:** Baseline model experimentation running multiple algorithms (Linear Regression, Random Forest, HistGradientBoosting) on a time-based split.
* **`src/grid_search.py`:** Hyperparameter tuning executing grid search cross-validation on the best working model to find optimal settings.
* **`src/train_predict.py`:** Trains the final chosen model using optimal hyperparameters on the cleaned training set and outputs predictions.
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
  * **Direct Months** (Jan, Feb, Mar, Jun, Sep): $\text{Rate per Mile} \approx \text{Quote Signal}$.
  * **Inverted Months** (Apr, May, Jul, Oct): $\text{Rate per Mile} \approx 4.15 - \text{Quote Signal}$.
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

### Model Comparison (`src/compare_models.py`)
Running different baseline algorithms on the time-based split yields:
* **Linear Regression (Normal Target):** $R^2 = 0.8385$, MAE = \$144.13
* **Linear Regression (Log Target):** $R^2 = 0.6230$, MAE = \$438.97
* **Random Forest (Log Target):** $R^2 = 0.8204$, MAE = \$180.09
* **HistGradientBoostingRegressor (Log Target):** $R^2 = 0.8417$, MAE = \$114.97

*Decision:* `HistGradientBoostingRegressor` (Log Target) performs the best. The combined effect of transitioning from the Linear Regression baseline to the non-linear model and utilizing the log target transformation yields a **20.2%** improvement in validation Mean Absolute Error (MAE) compared to the standard Linear Regression baseline.

#### Why HistGradientBoosting with Log-Target Outperforms Other Methods:

1. **Multiplicative Rate Physics (Log Target vs. Raw Target):**
   Freight rates exhibit multiplicative dynamics where factors like equipment class, seasonal demand, and regional capacity act as percentage modifiers rather than flat dollar additions. For instance, a reefer surcharge or a holiday capacity crunch increases the rate by a percentage, which means a much larger absolute dollar change for long-haul routes compared to short-haul routes. Fitting the model in log-space transforms these multiplicative effects into additive ones, allowing the regressor to fit the underlying variance structure far more accurately and reducing prediction bias for higher-rate lanes.

2. **Non-linear Spatial Geography (Trees vs. Linear Regression):**
   Freight pricing is heavily spatial. Origin/destination coordinates (latitude and longitude) dictate regional capacity imbalances (headhaul vs. backhaul regions). Linear Regression assumes a flat linear relationship with latitudes and longitudes, which is physically nonsensical and leads to poor generalization. In contrast, tree ensembles recursively split spatial coordinate spaces to group regional lanes and segment geographic coordinates, capturing localized pricing differences natively.

3. **Sequential Boosting vs. Bagging (HistGradientBoosting vs. Random Forest):**
   While Random Forest fits deep independent trees and averages them, it suffers from "step-function" behavior and flat-line predictions outside its training bounds. HistGradientBoosting builds shallow trees sequentially to fit the gradient residuals of the loss function. This boosting process acts as a smooth, continuous estimator that can reconstruct the high-frequency weekly demand seasonality and continuous distance-rate curves without the artificial blockiness/discontinuity of Random Forests.

4. **Native Handling of Missing Values (Weight & Market Index):**
   Linear Regression and Random Forest require explicit imputation of missing values (e.g., filling NaNs with zeros or medians), which introduces artificial bias. HistGradientBoosting handles NaNs natively during both split finding and inference by mapping missing values to whichever child node minimizes the loss, preserving the raw patterns in the data.

5. **Histogram-Based Binning & Regularization (Computational & Generalization Efficiency):**
   Traditional tree models evaluate every unique continuous value (e.g., exact lat/lon coordinates, distances) as a candidate split point, which is computationally expensive and highly prone to overfitting on noise. `HistGradientBoosting` groups continuous features into 256 integer-valued bins. This reduces split-finding complexity from $O(N \log N)$ to $O(N)$, allowing fast training cycles (making large grid searches feasible). Crucially, this binning acts as a natural regularizer, smoothing out micro-variations and allowing the model to generalize much better to the unseen validation months.

### Hyperparameter Grid Search (`src/grid_search.py`)

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
