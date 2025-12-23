# train_model_germany.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import sys
import io

# Set encoding untuk Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

def train_germany_model():
    print(">>> Training model untuk dataset Germany Cars...")
    
    try:
        # Load dataset Germany
        df = pd.read_csv('autoscout24-germany-dataset.csv')
        print(f"[OK] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("[ERROR] File 'autoscout24-germany-dataset.csv' tidak ditemukan!")
        print("[INFO] Silakan download dari: https://www.kaggle.com/datasets/ander289386/cars-germany")
        return False
    except Exception as e:
        print(f"[ERROR] Error loading dataset: {str(e)}")
        return False
    
    # Preprocessing khusus untuk dataset Germany
    print(">>> Preprocessing data...")
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Tampilkan kolom yang ada
    print(f"Kolom dataset: {list(df.columns)}")
    
    # Mapping kolom
    column_mapping = {
        'mileage': 'km_driven',
        'make': 'brand',
        'gear': 'transmission',
        'offertype': 'owner',
        'hp': 'horse_power',
        'year': 'registration_year'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
            print(f"  Renamed {old_col} -> {new_col}")
    
    # Pastikan kolom price ada
    if 'price' not in df.columns:
        print("[ERROR] Kolom 'price' tidak ditemukan dalam dataset!")
        print("  Kolom yang tersedia:", list(df.columns))
        return False
    
    # Tampilkan info dataset setelah preprocessing
    print(f"Dataset setelah preprocessing: {df.shape}")
    
    # Handle missing values
    print(">>> Handling missing values...")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            print(f"  {col}: {df[col].isnull().sum()} missing values")
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # Tampilkan statistik harga
    print(f"\nStatistik harga:")
    print(f"  Min: EUR {df['price'].min():,.0f}")
    print(f"  Max: EUR {df['price'].max():,.0f}")
    print(f"  Avg: EUR {df['price'].mean():,.0f}")
    print(f"  Std: EUR {df['price'].std():,.0f}")
    
    # Encoding categorical variables
    print("\n>>> Encoding categorical variables...")
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Hapus kolom dengan nilai unik terlalu banyak
    cols_to_encode = []
    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique <= 50:  # Hanya encode kolom dengan maksimal 50 nilai unik
            cols_to_encode.append(col)
            print(f"  {col}: {n_unique} unique values")
        else:
            print(f"  Skipping {col}: terlalu banyak nilai unik ({n_unique})")
            df = df.drop(columns=[col])
    
    for col in cols_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Pisahkan features dan target
    X = df.drop('price', axis=1)
    y = df['price']
    
    print(f"\nData untuk training:")
    print(f"  Features: {X.shape[1]} columns")
    print(f"  Samples: {X.shape[0]} rows")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain size: {X_train.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")
    
    # Scale features
    print("\n>>> Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\n>>> Training models...")
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        try:
            model.fit(X_train_scaled, y_train)
            
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            results[name] = {
                'model': model,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'y_pred_test': y_pred_test
            }
            
            print(f"    Train MAE: EUR {train_mae:,.0f}")
            print(f"    Test MAE: EUR {test_mae:,.0f}")
            print(f"    Train R2: {train_r2:.4f}")
            print(f"    Test R2: {test_r2:.4f}")
            
        except Exception as e:
            print(f"    Error training {name}: {str(e)}")
    
    if not results:
        print("[ERROR] Semua model gagal di-training!")
        return False
    
    # Pilih model terbaik
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\n[SUCCESS] Best model: {best_model_name}")
    print(f"  Test R2: {results[best_model_name]['test_r2']:.4f}")
    print(f"  Test MAE: EUR {results[best_model_name]['test_mae']:,.0f}")
    
    # Visualisasi hasil
    print("\n>>> Generating visualizations...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot
        axes[0, 0].scatter(y_test, results[best_model_name]['y_pred_test'], alpha=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()], 
                       [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Price (EUR)')
        axes[0, 0].set_ylabel('Predicted Price (EUR)')
        axes[0, 0].set_title(f'Actual vs Predicted - {best_model_name}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_test - results[best_model_name]['y_pred_test']
        axes[0, 1].scatter(results[best_model_name]['y_pred_test'], residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Price (EUR)')
        axes[0, 1].set_ylabel('Residuals (EUR)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution of residuals
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Residuals (EUR)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            axes[1, 1].barh(feature_importance['feature'].head(10)[::-1], 
                           feature_importance['importance'].head(10)[::-1])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Feature Importance')
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Importance')
        
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance_germany.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Performance plots saved as 'model_performance_germany.png'")
    except Exception as e:
        print(f"  Warning: Could not save plots: {str(e)}")
    
    # Save model
    print("\n>>> Saving model and preprocessing objects...")
    try:
        joblib.dump(best_model, 'model_germany.pkl')
        joblib.dump(scaler, 'scaler_germany.pkl')
        joblib.dump(label_encoders, 'label_encoders_germany.pkl')
        
        # Save feature names
        with open('feature_names_germany.txt', 'w', encoding='utf-8') as f:
            for feature in X.columns:
                f.write(f"{feature}\n")
        
        print("  [OK] Model berhasil disimpan!")
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Train_MAE': [results[m]['train_mae'] for m in results],
            'Test_MAE': [results[m]['test_mae'] for m in results],
            'Train_R2': [results[m]['train_r2'] for m in results],
            'Test_R2': [results[m]['test_r2'] for m in results]
        })
        
        results_df.to_csv('model_results_germany.csv', index=False)
        print("  [OK] Results saved as 'model_results_germany.csv'")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Error saving model: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("GERMANY CARS MODEL TRAINING SCRIPT")
    print("=" * 60)
    
    success = train_germany_model()
    
    if success:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Files created:")
        print("  - model_germany.pkl (model)")
        print("  - scaler_germany.pkl (scaler)")
        print("  - label_encoders_germany.pkl (encoders)")
        print("  - model_results_germany.csv (results)")
        print("  - model_performance_germany.png (plots)")
        print("  - feature_names_germany.txt (feature names)")
        print("\nYou can now run: streamlit run app.py")
    else:
        print("\n" + "=" * 60)
        print("TRAINING FAILED!")
        print("=" * 60)
        print("Tips:")
        print("  1. Pastikan file 'autoscout24-germany-dataset.csv' ada")
        print("  2. Download dari: kaggle.com/datasets/ander289386/cars-germany")
        print("  3. Periksa kolom 'price' ada dalam dataset")
        print("  4. Coba jalankan 'python create_test_data.py' untuk data test")