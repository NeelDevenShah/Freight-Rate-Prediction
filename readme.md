## 1. Key Data Findings & Exploratory Analysis

* **Tapering Effect of Distance:** There is a high correlation (~0.90) between `distance` and `posted_rate`. However, the rate per mile (`posted_rate / distance`) is non-linear; it tapers off as distance increases. Short-haul trips have a higher rate per mile due to fixed pickup/delivery costs, while long-haul trips have a lower rate per mile.
* **Equipment Class Pricing Profiles:** Rates vary systematically by equipment class:
  * **Dry Van:** Average rate per mile is the lowest (~$2.12/mile).
  * **Flatbed:** Average rate per mile is mid-range (~$2.29/mile).
  * **Reefer (Refrigerated):** Average rate per mile is the highest (~$2.38/mile).
* **Weekly Market Seasonality:** The `market_index` (and corresponding rates) follows a strong weekly cycle:
  * Rates and market activity bottom out on Sundays and Mondays.
  * Rates peak during mid-week (Wednesdays and Thursdays) due to increased shipping volume.

---

## 2. Data-Quality Issues & Mitigation Strategies

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

## 3. Modeling Approach & Reasoning

* **Chosen Model:** Scikit-Learn's `HistGradientBoostingRegressor` (a histogram-based gradient boosting tree ensemble).
* **Reasoning:**
  1. **Non-linear Relationship Modeling:** It excels at modeling non-linear logistics features (distance, weight, coordinates) and capture complex interactions (e.g., location interactions) without explicit feature transformation.
  2. **Native NaN and Categorical Handling:** It handles missing values natively and supports categorical variables (like equipment type and city codes).
  3. **Robustness to Noise:** Boosting models are highly robust to residual outliers in the target variable.
* **Target Log-Transformation:**
  * Freight rates scale multiplicatively with distance, weight, and market cycles.
  * We train the model to predict `log(posted_rate)` and then exponentiate the predictions (`exp(preds)`). 
  * Testing shows this log-transformation reduces Mean Absolute Error (MAE) by **~20%** (dropping from ~$121 to ~$97 on the validation split) compared to training on the raw target.
* **Optimal Hyperparameters:**
  * `learning_rate=0.03`
  * `max_iter=400`
  * `early_stopping=False` (prevents premature termination and ensures full convergence)

---

## 4. Training & Validation Strategy

* **Validation Split:** 
  * The training dataset (`data/train-test.csv`) was split into an 80% training set and a 20% validation set (randomly sampled) to evaluate local model performance and tune hyperparameters.
  * Additionally, time-based validation (using Month 10 as a validation set) was conducted to ensure the model generalizes to future periods.
* **Validation Performance Metrics (Local Validation Split):**
  * **$R^2$ Score:** **0.8703** (meaning the model explains 87% of the rate variance).
  * **Mean Absolute Error (MAE):** **$93.78** (on average, predictions are within ~$94 of the actual rate).

---

## 5. Code Walkthrough (`train_predict.py`)

The training and prediction pipeline is structured as follows:
1. **`haversine` & `preprocess_df`:** Parses the `date` column into `month`, `day`, `dayofweek`, `dayofyear`, and `is_weekend` features. Computes the great-circle Haversine distance and the ratio of routing distance to Haversine distance. Encodes categorical columns (`equipment`, `pickup`, `delivery`) using Scikit-Learn's `OrdinalEncoder`, handling unseen/unknown validation cities natively.
2. **Outlier Filtering:** Excludes training records where the rate per mile is $<0.5$ or $>10.0$.
3. **Training:** Fits the `HistGradientBoostingRegressor` on the log-transformed `posted_rate` using the engineered feature set.
4. **Validation Predictions:** Generates predictions for the validation dataset and saves them to `validation_predictions.csv` with exactly `load_id` and `predicted_rate` columns.
5. **December Chart Predictions:** Queries the coordinates of Lexington and Fort Wayne from the training data, imputes them into the December chart input records, runs the model, and fills the `predicted_rate` column of `data/december-chart-inputs.csv`.

---

## 6. How to Run the Code

### Step 1: Activate the environment
```bash
conda activate base
```

### Step 2: Execute training and predictions
```bash
python train_predict.py
```

### Step 3: Run the validation script
```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```
The script will validate the formats and generate the December load rate chart at `scorer_results/candidate_december.png`.
