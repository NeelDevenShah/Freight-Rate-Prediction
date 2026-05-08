import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

def haversine(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 3956 # Earth radius in miles

def preprocess_df(df, oe=None, fit_oe=False):
    df = df.copy()
    # Parse dates
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Categorical equipment
    df['equipment_cat'] = df['equipment'].astype('category').cat.codes
    
    # Haversine distance
    df['haversine_dist'] = haversine(df['pickup_lat'], df['pickup_lon'], df['delivery_lat'], df['delivery_lon'])
    df['dist_ratio'] = df['distance'] / (df['haversine_dist'] + 1e-5)
    
    # Ordinal encoding of pickup and delivery
    if fit_oe:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[['pickup_enc', 'delivery_enc']] = oe.fit_transform(df[['pickup', 'delivery']])
    else:
        df[['pickup_enc', 'delivery_enc']] = oe.transform(df[['pickup', 'delivery']])
        
    return df, oe

def main():
    print("Loading data...")
    df_train_raw = pd.read_csv('data/train-test.csv')
    df_val_raw = pd.read_csv('data/validation.csv')
    df_dec_raw = pd.read_csv('data/december-chart-inputs.csv')
    
    # Drop rows without targets in train
    df_train_raw = df_train_raw.dropna(subset=['posted_rate'])
    
    # Preprocess train
    print("Preprocessing training data...")
    df_train, oe = preprocess_df(df_train_raw, fit_oe=True)
    
    # Filter outliers in train (rate per mile < 0.5 or > 10.0)
    rpm = df_train['posted_rate'] / df_train['distance']
    clean_mask = (rpm >= 0.5) & (rpm <= 10.0)
    df_train_clean = df_train[clean_mask]
    print(f"Removed {len(df_train) - len(df_train_clean)} outliers from training set.")
    
    # Features for training
    features = [
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'distance', 'weight', 'equipment_cat',
        'month', 'day', 'dayofweek', 'dayofyear', 'is_weekend',
        'haversine_dist', 'dist_ratio', 'pickup_enc', 'delivery_enc'
    ]
    
    X_train = df_train_clean[features]
    y_train = np.log(df_train_clean['posted_rate'])
    
    print("Training HistGradientBoostingRegressor model...")
    # Using the optimal configuration that yields realistic seasonal variation
    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.03,
        early_stopping=False,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    # Preprocess validation
    print("Preprocessing validation data...")
    df_val, _ = preprocess_df(df_val_raw, oe=oe, fit_oe=False)
    X_val = df_val[features]
    
    # Predict validation rates
    print("Predicting validation set rates...")
    val_preds = np.exp(model.predict(X_val))
    
    # Fill in template format
    validation_predictions = pd.DataFrame({
        'load_id': df_val_raw['load_id'],
        'predicted_rate': val_preds
    })
    
    # Save validation predictions
    validation_predictions.to_csv('validation_predictions.csv', index=False)
    print("Saved validation predictions to validation_predictions.csv")
    
    # Preprocess December inputs
    print("Preprocessing December inputs...")
    # December input coordinates for Lexington to Fort Wayne
    # Lexington: lat 38.0406, lon -84.5007 (or we can lookup from train/val, let's verify exact values)
    # Let's extract coordinates of Lexington and Fort Wayne from training or validation data
    # to make sure they match perfectly
    lex_coords = df_train[df_train['pickup'] == 'Lexington'][['pickup_lat', 'pickup_lon']].iloc[0]
    fw_coords = df_train[df_train['delivery'] == 'Fort Wayne'][['delivery_lat', 'delivery_lon']].iloc[0]
    
    df_dec_filled = df_dec_raw.copy()
    df_dec_filled['pickup_lat'] = lex_coords['pickup_lat']
    df_dec_filled['pickup_lon'] = lex_coords['pickup_lon']
    df_dec_filled['delivery_lat'] = fw_coords['delivery_lat']
    df_dec_filled['delivery_lon'] = fw_coords['delivery_lon']
    
    # Preprocess December inputs
    df_dec_proc, _ = preprocess_df(df_dec_filled, oe=oe, fit_oe=False)
    X_dec = df_dec_proc[features]
    
    # Predict December rates
    print("Predicting December inputs...")
    dec_preds = np.exp(model.predict(X_dec))
    
    # Fill predictions in the dataframe and save back
    df_dec_raw['predicted_rate'] = dec_preds
    df_dec_raw.to_csv('data/december-chart-inputs.csv', index=False)
    print("Completed and saved December inputs to data/december-chart-inputs.csv")

if __name__ == '__main__':
    main()
