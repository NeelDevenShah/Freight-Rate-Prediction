import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import ParameterGrid

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 3956

def main():
    print("Loading data for grid search CV...")
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
    
    param_grid = {
        'target_log': [True, False],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_iter': [50, 100, 200, 400, 800]
    }
    
    results = []
    
    print("Running large grid search CV over 40 configurations...")
    for cfg in ParameterGrid(param_grid):
        model = HistGradientBoostingRegressor(
            learning_rate=cfg['learning_rate'],
            max_iter=cfg['max_iter'],
            early_stopping=False,
            random_state=42
        )
        
        # Train
        if cfg['target_log']:
            model.fit(X_train, np.log(y_train))
            preds = np.exp(model.predict(X_val))
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
        # Metrics
        score_r2 = r2_score(y_val, preds)
        score_mae = mean_absolute_error(y_val, preds)
        score_rmse = np.sqrt(mean_squared_error(y_val, preds))
        
        results.append({
            'target_log': cfg['target_log'],
            'learning_rate': cfg['learning_rate'],
            'max_iter': cfg['max_iter'],
            'r2': score_r2,
            'mae': score_mae,
            'rmse': score_rmse
        })
        
        print(f"  TargetLog={cfg['target_log']} lr={cfg['learning_rate']} iter={cfg['max_iter']} -> R2: {score_r2:.4f}, MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}")
        
    df_results = pd.DataFrame(results)
    df_results.to_csv('scorer_results/tuning_results.csv', index=False)
    print("\nTuning complete. Results stored in scorer_results/tuning_results.csv")
    
    # Print best config for MAE
    best_mae_row = df_results.loc[df_results['mae'].idxmin()]
    print(f"Best config by MAE: TargetLog={best_mae_row['target_log']} lr={best_mae_row['learning_rate']} iter={best_mae_row['max_iter']} -> MAE: {best_mae_row['mae']:.2f}")

if __name__ == '__main__':
    main()
