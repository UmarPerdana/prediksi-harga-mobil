# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath='used_cars.csv'):
    """
    Load dan preprocess dataset used cars
    """
    print("📂 Loading dataset...")
    
    try:
        # Load dataset
        df = pd.read_csv(filepath)
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("❌ File not found. Please download the dataset from Kaggle")
        print("📥 URL: https://www.kaggle.com/datasets/ananaymital/used-cars")
        return None, None, None
    
    # Tampilkan info dataset
    print("\n📊 Dataset Info:")
    print(df.info())
    
    print("\n🔍 First 5 rows:")
    print(df.head())
    
    # Analisis kolom
    print("\n📈 Column Analysis:")
    print("Columns:", df.columns.tolist())
    
    # Bersihkan nama kolom (hapus spasi, lowercase)
    df.columns = df.columns.str.strip().str.lower()
    
    # Cek kolom harga
    price_columns = ['price', 'selling_price', 'cost']
    price_col = None
    for col in price_columns:
        if col in df.columns:
            price_col = col
            break
    
    if price_col is None:
        print("❌ Price column not found!")
        # Coba cari kolom yang kemungkinan berisi harga
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].max() > 1000:
                price_col = col
                print(f"⚠️ Using '{col}' as price column")
                break
    
    if price_col:
        print(f"✅ Price column: {price_col}")
        df['price'] = df[price_col]
    
    # Drop kolom yang tidak perlu
    columns_to_drop = ['index', 's.no', 'id', 'car_id', 'unnamed: 0']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Identifikasi kolom kategorikal dan numerik
    categorical_cols = []
    numerical_cols = []
    
    for col in df.columns:
        if col == 'price':
            continue
        if df[col].dtype == 'object':
            categorical_cols.append(col)
        elif df[col].dtype in ['int64', 'float64']:
            numerical_cols.append(col)
    
    print(f"\n📌 Categorical columns: {categorical_cols}")
    print(f"📌 Numerical columns: {numerical_cols}")
    
    # Handle missing values
    print("\n🧹 Handling missing values...")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            print(f"   {col}: {df[col].isnull().sum()} missing values")
            if col in categorical_cols:
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # Handle outliers untuk kolom numerik
    print("\n📊 Handling outliers...")
    for col in numerical_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
    
    # Encoding kolom kategorikal
    print("\n🔤 Encoding categorical variables...")
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"   Encoded: {col} - {len(le.classes_)} unique values")
    
    # Pastikan kolom price ada
    if 'price' not in df.columns and len(df.columns) > 0:
        # Gunakan kolom numerik terakhir sebagai target
        last_num_col = numerical_cols[-1] if numerical_cols else df.columns[-1]
        df['price'] = df[last_num_col]
        df = df.drop(columns=[last_num_col])
        print(f"⚠️ Created 'price' column from '{last_num_col}'")
    
    # Hapus kolom dengan nilai unik terlalu banyak atau terlalu sedikit
    columns_to_keep = []
    for col in df.columns:
        if col == 'price':
            columns_to_keep.append(col)
            continue
        n_unique = df[col].nunique()
        if 1 < n_unique < 100:  # Ubah threshold sesuai kebutuhan
            columns_to_keep.append(col)
        else:
            print(f"   Dropped {col}: {n_unique} unique values")
    
    df = df[columns_to_keep]
    
    # Scale features
    print("\n⚖️ Scaling features...")
    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col != 'price']
    
    if feature_cols:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        print(f"   Scaled {len(feature_cols)} features")
    else:
        print("⚠️ No feature columns to scale!")
    
    print(f"\n✅ Preprocessing completed!")
    print(f"   Final shape: {df.shape}")
    print(f"   Features: {len(feature_cols)}")
    
    return df, label_encoders, scaler

if __name__ == "__main__":
    df, encoders, scaler = load_and_preprocess_data()
    if df is not None:
        print("\n💾 Saving processed data...")
        df.to_csv('processed_cars.csv', index=False)
        joblib.dump(encoders, 'label_encoders.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        print("✅ Data saved as 'processed_cars.csv'")