import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 3956

def main():
    print("Loading data for model evaluation...")
    df = pd.read_csv('data/train-test.csv').dropna(subset=['posted_rate'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Feature engineering
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[['pickup_enc', 'delivery_enc']] = oe.fit_transform(df[['pickup', 'delivery']])
    df['equipment_cat'] = df['equipment'].astype('category').cat.codes
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['haversine_dist'] = haversine(df['pickup_lat'], df['pickup_lon'], df['delivery_lat'], df['delivery_lon'])
    df['dist_ratio'] = df['distance'] / (df['haversine_dist'] + 1e-5)
    
    features = [
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'distance', 'weight', 'equipment_cat',
        'month', 'day', 'dayofweek', 'dayofyear', 'is_weekend',
        'haversine_dist', 'dist_ratio', 'pickup_enc', 'delivery_enc'
    ]
    
    # Filter outliers
    rpm = df['posted_rate'] / df['distance']
    df_clean = df[(rpm >= 0.5) & (rpm <= 10.0)]
    
    # Time-based split: train months 1-9, validate month 10
    train_idx = df_clean['month'] < 10
    val_idx = df_clean['month'] == 10
    
    X_train = df_clean.loc[train_idx, features]
    y_train = df_clean.loc[train_idx, 'posted_rate']
    X_val = df_clean.loc[val_idx, features]
    y_val = df_clean.loc[val_idx, 'posted_rate']
    
    # Fill NaNs for baseline models
    X_train_filled = X_train.fillna(0)
    X_val_filled = X_val.fillna(0)
    
    print("Running baseline models:")
    
    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_filled, y_train)
    preds_lr = lr.predict(X_val_filled)
    print(f"  Linear Regression (Normal) -> R2: {r2_score(y_val, preds_lr):.4f}, MAE: {mean_absolute_error(y_val, preds_lr):.2f}")
    
    # 2. Linear Regression (Log Target)
    lr_log = LinearRegression()
    lr_log.fit(X_train_filled, np.log(y_train))
    preds_lr_log = np.exp(lr_log.predict(X_val_filled))
    print(f"  Linear Regression (Log)    -> R2: {r2_score(y_val, preds_lr_log):.4f}, MAE: {mean_absolute_error(y_val, preds_lr_log):.2f}")
    
    # 3. Random Forest (Log Target)
    rf_log = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    rf_log.fit(X_train_filled, np.log(y_train))
    preds_rf_log = np.exp(rf_log.predict(X_val_filled))
    print(f"  Random Forest (Log)        -> R2: {r2_score(y_val, preds_rf_log):.4f}, MAE: {mean_absolute_error(y_val, preds_rf_log):.2f}")
    
    # 4. HistGradientBoosting (Log Target)
    hgb_log = HistGradientBoostingRegressor(random_state=42)
    hgb_log.fit(X_train, np.log(y_train))
    preds_hgb_log = np.exp(hgb_log.predict(X_val))
    print(f"  HistGradientBoosting (Log) -> R2: {r2_score(y_val, preds_hgb_log):.4f}, MAE: {mean_absolute_error(y_val, preds_hgb_log):.2f}")
    print()

if __name__ == '__main__':
    main()
