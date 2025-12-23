# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Prediksi Harga Mobil Bekas (Germany)",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 2rem;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Title - DIUBAH untuk mencerminkan dataset baru
st.markdown('<h1 class="main-header">🚗 Prediksi Harga Mobil Bekas (Germany)</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #6B7280; margin-bottom: 2rem;'>
Aplikasi ini menggunakan Machine Learning untuk memprediksi harga mobil bekas di Jerman berdasarkan berbagai fitur.
Dataset dari <a href='https://www.kaggle.com/datasets/ander289386/cars-germany' target='_blank'>Germany Cars Dataset</a>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2489/2489227.png", width=100)
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Dashboard", "📊 EDA", "🤖 Training", "🔮 Prediksi", "📈 Evaluasi"]
)

# Load data - DIUBAH untuk dataset Germany Cars
@st.cache_data
def load_data():
    try:
        # Menggunakan dataset Germany Cars
        df = pd.read_csv('autoscout24-germany-dataset.csv')
        st.sidebar.success("✅ Dataset Germany Cars loaded")
        
        # Menampilkan informasi dataset
        st.sidebar.info(f"📊 Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Membersihkan nama kolom (menghapus spasi, lowercase)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # DIUBAH: Mapping kolom dataset Germany Cars
        # Dataset memiliki kolom: mileage, make, model, fuel, gear, offertype, price, hp, year
        # Rename kolom untuk konsistensi dengan kode
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
        
        # DIUBAH: Membuat kolom 'price' sebagai target (jika belum ada)
        if 'price' in df.columns:
            df['price'] = df['price']
        else:
            st.sidebar.warning("⚠️ Kolom 'price' tidak ditemukan dalam dataset")
        
        return df
        
    except FileNotFoundError:
        st.sidebar.error("❌ File 'autoscout24-germany-dataset.csv' tidak ditemukan")
        st.sidebar.markdown("""
        **Silakan download dataset dari:**
        https://www.kaggle.com/datasets/ander289386/cars-germany
        
        **Dan simpan sebagai:** `autoscout24-germany-dataset.csv`
        """)
        
        # Fallback ke data sample
        st.sidebar.warning("⚠️ Menggunakan data sample untuk demo")
        return create_sample_data()
    except Exception as e:
        st.sidebar.error(f"❌ Error loading dataset: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Membuat data sample untuk demo"""
    np.random.seed(42)
    n_samples = 500
    
    # DIUBAH: Menyesuaikan dengan dataset Germany Cars
    data = {
        'brand': np.random.choice(['Volkswagen', 'Opel', 'BMW', 'Mercedes', 'Audi'], n_samples),
        'model': np.random.choice(['Golf', 'Corsa', '320i', 'C200', 'A4'], n_samples),
        'registration_year': np.random.randint(2011, 2021, n_samples),
        'km_driven': np.random.randint(1000, 200000, n_samples),
        'fuel': np.random.choice(['Gasoline', 'Diesel', 'Electric', 'Hybrid'], n_samples),
        'transmission': np.random.choice(['Manual', 'Automatic', 'Semi-Automatic'], n_samples),
        'owner': np.random.choice(['Used', 'Pre-registered', 'Demonstration', 'Employee'], n_samples),
        'horse_power': np.random.randint(50, 300, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate price based on Germany car market
    df['price'] = (
        20000 +  # Base price
        (df['registration_year'] - 2010) * 1000 +  # Newer = more expensive
        (200000 - df['km_driven']) * 0.15 +  # Lower km = higher price
        np.where(df['fuel'] == 'Diesel', 3000, 0) +
        np.where(df['transmission'] == 'Automatic', 4000, 0) +
        df['horse_power'] * 50 +  # More HP = more expensive
        np.where(df['brand'] == 'Mercedes', 10000, 
                np.where(df['brand'] == 'BMW', 8000, 
                        np.where(df['brand'] == 'Audi', 7000, 0))) +
        np.random.normal(0, 5000, n_samples)  # Random noise
    )
    
    # Ensure price is positive
    df['price'] = df['price'].clip(lower=5000)
    
    return df

df = load_data()

# Dashboard Page - DIUBAH untuk menampilkan info dataset Germany Cars
# Dashboard Page - DIUBAH untuk menampilkan info dataset Germany Cars
if page == "🏠 Dashboard":
    # CSS tambahan untuk dashboard yang lebih baik
    st.markdown("""
    <style>
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-card-improved {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            border-left: 5px solid #667eea;
            transition: transform 0.3s ease;
        }
        .metric-card-improved:hover {
            transform: translateY(-5px);
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #2d3748;
            margin: 0.5rem 0;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            color: #667eea;
        }
        .data-summary-card {
            background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header Dashboard yang lebih menarik
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">🚗 Dashboard Data Mobil Jerman</h1>
        <p style="font-size: 1.1rem; opacity: 0.9;">
            Analisis komprehensif dataset mobil bekas Jerman dengan 46,405 records dari AutoScout24
        </p>
        <div style="background: rgba(255, 255, 255, 0.2); padding: 0.5rem 1rem; border-radius: 8px; display: inline-block; margin-top: 1rem;">
            <span style="font-size: 0.9rem;">📅 Data terbaru | 💰 Harga dalam Euro | 🇩🇪 Market Germany</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Section dengan layout yang lebih baik
    st.markdown('<h2 style="color: #2d3748; margin-bottom: 1.5rem;">📊 Ringkasan Statistik</h2>', unsafe_allow_html=True)
    
    # Row 1: Metrics utama
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card-improved">
            <div class="metric-icon">📊</div>
            <div class="metric-label">Total Data</div>
            <div class="metric-value">{:,}</div>
            <div style="font-size: 0.85rem; color: #718096;">Records mobil bekas</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        avg_price = df['price'].mean() if 'price' in df.columns else 0
        st.markdown("""
        <div class="metric-card-improved">
            <div class="metric-icon">💰</div>
            <div class="metric-label">Harga Rata-rata</div>
            <div class="metric-value">€{:,.0f}</div>
            <div style="font-size: 0.85rem; color: #718096;">Market value rata-rata</div>
        </div>
        """.format(avg_price), unsafe_allow_html=True)
    
    with col3:
        year_col = 'registration_year' if 'registration_year' in df.columns else 'year'
        avg_year = df[year_col].mean() if year_col in df.columns else 0
        st.markdown("""
        <div class="metric-card-improved">
            <div class="metric-icon">📅</div>
            <div class="metric-label">Tahun Rata-rata</div>
            <div class="metric-value">{:.0f}</div>
            <div style="font-size: 0.85rem; color: #718096;">Tahun registrasi</div>
        </div>
        """.format(avg_year), unsafe_allow_html=True)
    
    with col4:
        if 'horse_power' in df.columns:
            avg_hp = df['horse_power'].mean() if 'horse_power' in df.columns else 0
            st.markdown("""
            <div class="metric-card-improved">
                <div class="metric-icon">⚡</div>
                <div class="metric-label">Daya Rata-rata</div>
                <div class="metric-value">{:.1f}</div>
                <div style="font-size: 0.85rem; color: #718096;">Horse Power (HP)</div>
            </div>
            """.format(avg_hp), unsafe_allow_html=True)
        else:
            km_col = 'km_driven' if 'km_driven' in df.columns else 'mileage'
            avg_km = df[km_col].mean() if km_col in df.columns else 0
            st.markdown("""
            <div class="metric-card-improved">
                <div class="metric-icon">🛣️</div>
                <div class="metric-label">Km Rata-rata</div>
                <div class="metric-value">{:,.0f}</div>
                <div style="font-size: 0.85rem; color: #718096;">Kilometers driven</div>
            </div>
            """.format(avg_km), unsafe_allow_html=True)
    
    # Row 2: Informasi tambahan
    st.markdown('<h2 style="color: #2d3748; margin: 2rem 0 1.5rem 0;">📈 Informasi Tambahan</h2>', unsafe_allow_html=True)
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        if 'brand' in df.columns:
            unique_brands = df['brand'].nunique()
            st.markdown(f"""
            <div class="data-summary-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">Merek Unik</div>
                <div style="font-size: 2.5rem; font-weight: 800;">{unique_brands}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Jumlah merek mobil berbeda</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col6:
        if 'fuel' in df.columns:
            fuel_types = df['fuel'].nunique()
            st.markdown(f"""
            <div class="data-summary-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">Tipe Bahan Bakar</div>
                <div style="font-size: 2.5rem; font-weight: 800;">{fuel_types}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Jenis bahan bakar berbeda</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col7:
        if 'price' in df.columns:
            max_price = df['price'].max()
            st.markdown(f"""
            <div class="data-summary-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">Harga Tertinggi</div>
                <div style="font-size: 1.8rem; font-weight: 800;">€{max_price:,.0f}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Mobil termahal dalam dataset</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Data Preview dengan tab yang lebih baik
    st.markdown('<h2 style="color: #2d3748; margin: 2rem 0 1.5rem 0;">📋 Preview Data (Germany Cars Dataset)</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Data Sample", "📈 Quick Stats", "🔍 Column Info"])
    
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Menampilkan 10 dari {len(df)} total records. Harga dalam Euro (€)")
    
    with tab2:
        if 'price' in df.columns:
            # Quick statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Harga Minimum", f"€{df['price'].min():,.0f}")
            with col2:
                st.metric("Harga Median", f"€{df['price'].median():,.0f}")
            with col3:
                st.metric("Harga Maksimum", f"€{df['price'].max():,.0f}")
            
            # Tampilkan distribusi singkat
            st.subheader("Distribusi Harga Singkat")
            fig, ax = plt.subplots(figsize=(10, 4))
            df['price'].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Harga (€)')
            ax.set_ylabel('Frekuensi')
            ax.set_title('Distribusi Harga Mobil Jerman')
            st.pyplot(fig)
    
    with tab3:
        st.write("**Informasi Kolom Dataset:**")
        column_info = pd.DataFrame({
            'Kolom': df.columns,
            'Tipe': df.dtypes.values,
            'Non-Null': df.notnull().sum().values,
            'Unique': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(column_info, use_container_width=True)
    
    # Informasi dataset Germany Cars
    st.markdown('<h2 style="color: #2d3748; margin: 2rem 0 1.5rem 0;">ℹ️ Karakteristik Dataset Germany Cars</h2>', unsafe_allow_html=True)
    
    if 'brand' in df.columns:
        # Top 5 brands
        col1, col2 = st.columns([2, 1])
        
        with col1:
            top_brands = df['brand'].value_counts().head(5)
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.Set3(np.arange(len(top_brands)))
            ax.barh(top_brands.index, top_brands.values, color=colors)
            ax.set_xlabel('Jumlah Mobil')
            ax.set_title('5 Merek Mobil Terbanyak (Germany Market)')
            for i, v in enumerate(top_brands.values):
                ax.text(v + 10, i, str(v), color='black', fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                <h4 style="color: #2d3748; margin-top: 0;">📌 Insight Market Germany</h4>
                <p style="font-size: 0.95rem;">
                <strong>Karakteristik unik mobil Jerman:</strong>
                <ul style="font-size: 0.9rem;">
                    <li>Merek lokal dominan (VW, BMW, Mercedes)</li>
                    <li>Diesel masih populer</li>
                    <li>Transmisi manual lebih umum</li>
                    <li>Umur mobil rata-rata 4-6 tahun</li>
                </ul>
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tampilkan beberapa contoh data dengan harga menarik
    st.markdown('<h3 style="color: #2d3748; margin: 2rem 0 1rem 0;">🚀 Contoh Mobil dengan Spesifikasi Menarik</h3>', unsafe_allow_html=True)
    
    if all(col in df.columns for col in ['brand', 'model', 'price', 'registration_year']):
        # Mobil termurah
        cheapest = df.nsmallest(3, 'price')
        # Mobil termahal
        expensive = df.nlargest(3, 'price')
        # Mobil dengan HP tertinggi
        if 'horse_power' in df.columns:
            powerful = df.nlargest(3, 'horse_power')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("💰 Termurah")
            for i, row in cheapest.iterrows():
                st.write(f"**{row['brand']} {row.get('model', '')}**")
                st.write(f"Tahun: {row['registration_year']} | Harga: €{row['price']:,.0f}")
                st.write("---")
        
        with col2:
            st.subheader("🏎️ Termahal")
            for i, row in expensive.iterrows():
                st.write(f"**{row['brand']} {row.get('model', '')}**")
                st.write(f"Tahun: {row['registration_year']} | Harga: €{row['price']:,.0f}")
                st.write("---")
        
        with col3:
            if 'horse_power' in df.columns:
                st.subheader("⚡ Terkuat")
                for i, row in powerful.iterrows():
                    st.write(f"**{row['brand']} {row.get('model', '')}**")
                    st.write(f"HP: {row['horse_power']} | Harga: €{row['price']:,.0f}")
                    st.write("---")

# EDA Page - DIUBAH untuk dataset Germany Cars
elif page == "📊 EDA":
    st.markdown('<h2 class="sub-header">📈 Exploratory Data Analysis (Germany Cars)</h2>', unsafe_allow_html=True)
    
    if 'price' in df.columns:
        # Distribution plots - DIUBAH untuk Euro
        st.subheader("Distribusi Harga (€)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['price'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            ax.set_xlabel('Harga (€)')
            ax.set_ylabel('Frekuensi')
            ax.set_title('Distribusi Harga Mobil (Germany)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(df['price'])
            ax.set_ylabel('Harga (€)')
            ax.set_title('Boxplot Harga (Germany)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Scatter plots - DIUBAH untuk fitur Germany Cars
        st.subheader("Hubungan Fitur dengan Harga")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year_col = 'registration_year' if 'registration_year' in df.columns else 'year'
            if year_col in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(df[year_col], df['price'], alpha=0.5)
                ax.set_xlabel('Tahun Registrasi')
                ax.set_ylabel('Harga (€)')
                ax.set_title('Harga vs Tahun Registrasi')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        with col2:
            km_col = 'km_driven' if 'km_driven' in df.columns else 'mileage'
            if km_col in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(df[km_col], df['price'], alpha=0.5, color='green')
                ax.set_xlabel('Kilometer')
                ax.set_ylabel('Harga (€)')
                ax.set_title('Harga vs Kilometer')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        # Horse power analysis (khusus Germany dataset)
        if 'horse_power' in df.columns:
            st.subheader("Analisis Horse Power")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df['horse_power'], df['price'], alpha=0.5, color='red')
            ax.set_xlabel('Horse Power (HP)')
            ax.set_ylabel('Harga (€)')
            ax.set_title('Harga vs Horse Power')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Categorical analysis - DIUBAH untuk dataset Germany
        st.subheader("Analisis Kategorikal")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            selected_cat = st.selectbox("Pilih kolom kategorikal:", categorical_cols[:5])
            
            if selected_cat in df.columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                price_by_cat = df.groupby(selected_cat)['price'].mean().sort_values(ascending=False)
                price_by_cat.head(10).plot(kind='bar', ax=ax, color='lightcoral')
                ax.set_xlabel(selected_cat)
                ax.set_ylabel('Rata-rata Harga (€)')
                ax.set_title(f'Rata-rata Harga berdasarkan {selected_cat} (Top 10)')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

# Training Page - TIDAK DIUBAH (tetap kompatibel)
elif page == "🤖 Training":
    st.markdown('<h2 class="sub-header">🤖 Training Model</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Langkah-langkah Training:**
    1. Preprocessing data untuk dataset Germany Cars
    2. Split data training & testing
    3. Train model dengan algoritma berbeda
    4. Evaluasi dan pilih model terbaik
    
    **Note:** Dataset yang digunakan: Germany Cars Dataset
    """)
    
    if st.button("🚀 Mulai Training", type="primary"):
        with st.spinner("Training model dengan dataset Germany Cars..."):
            try:
                # Import training script
                import subprocess
                import sys
                
                # Run training script
                result = subprocess.run([sys.executable, "train_model_germany.py"], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.success("✅ Training selesai!")
                    
                    # Show output
                    st.text_area("Output Training:", result.stdout, height=300)
                    
                    # Load and show results
                    try:
                        results_df = pd.read_csv('model_results_germany.csv')
                        st.subheader("📊 Hasil Perbandingan Model")
                        st.dataframe(results_df.style.highlight_max(subset=['Test_R2']), 
                                   use_container_width=True)
                        
                        # Show plots
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image('model_performance_germany.png', 
                                   caption='Model Performance (Germany)')
                        with col2:
                            st.image('feature_importance_germany.png', 
                                   caption='Feature Importance (Germany)')
                    
                    except:
                        st.warning("File hasil tidak ditemukan. Pastikan training script berjalan dengan benar.")
                
                else:
                    st.error("❌ Training gagal!")
                    st.text("Error output:")
                    st.text(result.stderr)
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Coba buat file 'train_model_germany.py' untuk dataset Germany Cars")

# Prediction Page - DIUBAH untuk dataset Germany Cars
# Prediction Page - DIUBAH untuk dataset Germany Cars (FIXED VERSION)
elif page == "🔮 Prediksi":
    st.markdown('<h2 class="sub-header">🔮 Prediksi Harga Mobil (Germany)</h2>', unsafe_allow_html=True)
    
    # Load model dengan error handling yang lebih baik
    model_loaded = False
    
    try:
        model = joblib.load('model_germany.pkl')
        scaler = joblib.load('scaler_germany.pkl')
        label_encoders = joblib.load('label_encoders_germany.pkl')
        
        # Coba load feature names dari file
        try:
            with open('feature_names_germany.txt', 'r') as f:
                training_features = [line.strip() for line in f.readlines()]
        except:
            # Jika file tidak ada, coba ambil dari scaler atau infer dari dataset
            if hasattr(scaler, 'feature_names_in_'):
                training_features = list(scaler.feature_names_in_)
            else:
                # Fallback: ambil semua kolom kecuali price
                training_features = [col for col in df.columns if col != 'price' and col in df.columns]
        
        st.sidebar.success(f"✅ Model loaded ({len(training_features)} features)")
        model_loaded = True
        
    except Exception as e:
        st.warning(f"⚠️ Model belum ditraining atau error: {str(e)}")
        model_loaded = False
    
    # Input form - DIUBAH untuk fitur Germany Cars
    st.subheader("Masukkan Data Mobil (Germany Market)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Brand selection
        if 'brand' in df.columns:
            available_brands = list(df['brand'].dropna().unique())[:20]
            brand = st.selectbox("Merek", available_brands)
        else:
            brand = st.text_input("Merek", "Volkswagen")
        
        # Year selection
        year_col = 'registration_year' if 'registration_year' in df.columns else 'year'
        if year_col in df.columns:
            min_year = int(df[year_col].min()) if len(df[year_col].dropna()) > 0 else 2010
            max_year = int(df[year_col].max()) if len(df[year_col].dropna()) > 0 else 2023
            year = st.slider("Tahun Registrasi", min_year, max_year, 2018)
        else:
            year = st.number_input("Tahun Registrasi", 2000, 2023, 2018)
        
        # KM driven
        km_col = 'km_driven' if 'km_driven' in df.columns else 'mileage'
        if km_col in df.columns and len(df[km_col].dropna()) > 0:
            max_km = int(df[km_col].max())
            km_driven = st.number_input("Kilometer", 0, max_km, 50000, step=1000)
        else:
            km_driven = st.number_input("Kilometer", 0, 500000, 50000)
    
    with col2:
        # Fuel type
        if 'fuel' in df.columns and len(df['fuel'].dropna()) > 0:
            fuel_options = list(df['fuel'].dropna().unique())
            fuel = st.selectbox("Bahan Bakar", fuel_options)
        else:
            fuel = st.selectbox("Bahan Bakar", ['Gasoline', 'Diesel', 'Electric', 'Hybrid'])
        
        # Transmission
        if 'transmission' in df.columns and len(df['transmission'].dropna()) > 0:
            transmission_options = list(df['transmission'].dropna().unique())
            transmission = st.selectbox("Transmisi", transmission_options)
        else:
            transmission = st.selectbox("Transmisi", ['Manual', 'Automatic', 'Semi-Automatic'])
        
        # Owner/Offer type
        if 'owner' in df.columns and len(df['owner'].dropna()) > 0:
            owner_options = list(df['owner'].dropna().unique())
            owner = st.selectbox("Tipe Penawaran", owner_options)
        else:
            owner = st.selectbox("Tipe Penawaran", ['Used', 'Pre-registered', 'Demonstration'])
    
    # Additional features - DIUBAH untuk Germany Cars
    col3, col4 = st.columns(2)
    with col3:
        # Horse power
        if 'horse_power' in df.columns and len(df['horse_power'].dropna()) > 0:
            min_hp = int(df['horse_power'].min())
            max_hp = int(df['horse_power'].max())
            horse_power = st.slider("Horse Power (HP)", min_hp, max_hp, 120)
        else:
            horse_power = st.number_input("Horse Power (HP)", 50, 500, 120)
        
        # Model selection (opsional - hanya jika 'model' ada dalam training features)
        model_car = None
        if 'model' in df.columns and model_loaded and 'model' in training_features:
            # Filter models based on selected brand
            if brand and brand in df['brand'].values:
                brand_models = df[df['brand'] == brand]['model'].dropna().unique()[:10]
                if len(brand_models) > 0:
                    model_car = st.selectbox("Model (Opsional)", ["-- Pilih Model --"] + list(brand_models))
                    if model_car == "-- Pilih Model --":
                        model_car = None
    
    with col4:
        # Tampilkan info fitur model jika sudah loaded
        if model_loaded:
            with st.expander("📊 Fitur yang digunakan model"):
                st.write(f"**Jumlah fitur:** {len(training_features)}")
                st.write("**Daftar fitur:**")
                for i, feat in enumerate(training_features[:15], 1):
                    st.write(f"{i}. {feat}")
                if len(training_features) > 15:
                    st.write(f"... dan {len(training_features) - 15} fitur lainnya")
        else:
            st.info("ℹ️ Training model terlebih dahulu untuk melihat fitur")
    
    if st.button("💰 Prediksi Harga", type="primary", key="predict_button"):
        if not model_loaded:
            st.error("Model tidak tersedia. Silakan training model terlebih dahulu di halaman 'Training'.")
        else:
            with st.spinner("Memprediksi harga mobil Jerman..."):
                try:
                    # 1. Siapkan dictionary untuk semua nilai fitur
                    input_dict = {}
                    
                    # Mapping antara input user dan nama fitur
                    feature_mapping = {
                        'brand': brand,
                        'registration_year': year,
                        'km_driven': km_driven,
                        'fuel': fuel,
                        'transmission': transmission,
                        'owner': owner,
                        'horse_power': horse_power
                    }
                    
                    # Tambahkan model jika dipilih
                    if model_car and model_car != "-- Pilih Model --":
                        feature_mapping['model'] = model_car
                    
                    # 2. Untuk setiap fitur yang digunakan dalam training, isi nilainya
                    for feature in training_features:
                        if feature in feature_mapping:
                            # Jika fitur ada di input user
                            input_dict[feature] = feature_mapping[feature]
                        else:
                            # Jika fitur tidak ada di input user, beri nilai default
                            if feature in df.columns:
                                if df[feature].dtype == 'object':
                                    # Untuk kategorikal, gunakan nilai paling sering
                                    input_dict[feature] = df[feature].mode()[0] if len(df[feature].mode()) > 0 else 'Unknown'
                                else:
                                    # Untuk numerik, gunakan median
                                    input_dict[feature] = float(df[feature].median())
                            else:
                                # Jika fitur tidak ada di dataset sama sekali
                                input_dict[feature] = 0
                    
                    # 3. Buat DataFrame dengan urutan yang PERSIS sama dengan training
                    input_df = pd.DataFrame([input_dict])
                    input_df = input_df[training_features]  # Urutkan kolom
                    
                    # 4. Encode variabel kategorikal
                    for col in training_features:
                        if col in label_encoders:
                            try:
                                # Cek apakah nilai ada dalam encoder
                                value = str(input_df[col].iloc[0])
                                if value in label_encoders[col].classes_:
                                    input_df[col] = label_encoders[col].transform([value])[0]
                                else:
                                    # Jika nilai tidak dikenal, gunakan nilai default (0)
                                    st.warning(f"Nilai '{value}' tidak ditemukan dalam training untuk '{col}'. Menggunakan nilai default.")
                                    input_df[col] = 0
                            except Exception as e:
                                st.error(f"Error encoding {col}: {str(e)}")
                                input_df[col] = 0
                    
                    # 5. Debug: tampilkan data sebelum scaling
                    with st.expander("🔍 Data sebelum scaling (Debug)"):
                        st.write("**Input DataFrame:**")
                        st.dataframe(input_df)
                        st.write(f"**Shape:** {input_df.shape}")
                        st.write(f"**Columns:** {list(input_df.columns)}")
                    
                    # 6. Scale features
                    try:
                        input_scaled = scaler.transform(input_df)
                        
                        # 7. Predict
                        prediction = model.predict(input_scaled)[0]
                        
                        # 8. Tampilkan hasil
                        st.success(f"### 🎯 Prediksi Harga: **€{prediction:,.2f}**")
                        
                        # Confidence interval
                        lower_bound = max(0, prediction * 0.85)  # Pastikan tidak negatif
                        upper_bound = prediction * 1.15
                        
                        st.info(f"""
                        **Estimasi Rentang Harga:** €{lower_bound:,.0f} - €{upper_bound:,.0f}
                        
                        *Catatan: Prediksi ini memiliki margin error sekitar ±15%. 
                        Harga aktual dapat bervariasi tergantung kondisi mobil.*
                        """)
                        
                        # 9. Tampilkan ringkasan input
                        with st.expander("📋 Ringkasan Input"):
                            st.json(feature_mapping)
                        
                        # 10. Tampilkan nilai fitur setelah preprocessing
                        with st.expander("🔧 Nilai Fitur Setelah Preprocessing"):
                            display_df = pd.DataFrame({
                                'Fitur': training_features,
                                'Nilai Asli': [input_dict[feat] for feat in training_features],
                                'Nilai Encoded': input_df.iloc[0].values,
                                'Tipe': ['Kategorikal' if feat in label_encoders else 'Numerik' for feat in training_features]
                            })
                            st.dataframe(display_df)
                        
                        # 11. Berikan saran berdasarkan prediksi
                        with st.expander("💡 Saran dan Informasi"):
                            if prediction < 10000:
                                st.write("**Kategori:** Mobil ekonomis")
                                st.write("**Saran:** Cocok untuk penggunaan sehari-hari dengan budget terbatas")
                            elif prediction < 25000:
                                st.write("**Kategori:** Mobil menengah")
                                st.write("**Saran:** Keseimbangan baik antara harga dan fitur")
                            elif prediction < 50000:
                                st.write("**Kategori:** Mobil premium")
                                st.write("**Saran:** Cocok untuk yang mengutamakan kenyamanan dan performa")
                            else:
                                st.write("**Kategori:** Mobil mewah")
                                st.write("**Saran:** Investasi untuk kendaraan dengan spesifikasi tinggi")
                    
                    except Exception as e:
                        st.error(f"❌ Error dalam scaling atau prediksi: {str(e)}")
                        st.info("""
                        **Kemungkinan penyebab:**
                        1. Data input tidak lengkap
                        2. Format data tidak sesuai
                        3. Model rusak atau tidak kompatibel
                        
                        **Solusi:** Coba training ulang model di halaman 'Training'
                        """)
                        
                        # Tampilkan info debug
                        with st.expander("🐛 Debug Info"):
                            st.write(f"Input shape: {input_df.shape}")
                            st.write(f"Scaler expects: {scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 'Unknown'}")
                            st.write(f"Model expects: {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'Unknown'}")
                
                except Exception as e:
                    st.error(f"❌ Error dalam mempersiapkan data: {str(e)}")
                    st.info("Pastikan semua input telah diisi dengan benar.")

# Evaluation Page - TIDAK DIUBAH (tetap kompatibel)
elif page == "📈 Evaluasi":
    st.markdown('<h2 class="sub-header">📈 Evaluasi Model (Germany Cars)</h2>', unsafe_allow_html=True)
    
    try:
        # Coba load results untuk Germany dataset
        results_df = pd.read_csv('model_results_germany.csv')
        
        st.subheader("Perbandingan Model (Germany Dataset)")
        
        # Format metrics
        display_df = results_df.copy()
        display_df['Train_MAE'] = display_df['Train_MAE'].apply(lambda x: f"€{x:,.0f}")
        display_df['Test_MAE'] = display_df['Test_MAE'].apply(lambda x: f"€{x:,.0f}")
        display_df['Train_R2'] = display_df['Train_R2'].apply(lambda x: f"{x:.4f}")
        display_df['Test_R2'] = display_df['Test_R2'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Visual comparison
        st.subheader("Visualisasi Performa Model")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MAE comparison
        axes[0, 0].bar(results_df['Model'], results_df['Train_MAE'], alpha=0.7, label='Train')
        axes[0, 0].bar(results_df['Model'], results_df['Test_MAE'], alpha=0.7, label='Test')
        axes[0, 0].set_ylabel('MAE (€)')
        axes[0, 0].set_title('Mean Absolute Error Comparison')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # R² comparison
        axes[0, 1].bar(results_df['Model'], results_df['Train_R2'], alpha=0.7, label='Train')
        axes[0, 1].bar(results_df['Model'], results_df['Test_R2'], alpha=0.7, label='Test')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('R² Score Comparison')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Test metrics
        axes[1, 0].scatter(results_df['Test_MAE'], results_df['Test_R2'], s=200)
        for i, row in results_df.iterrows():
            axes[1, 0].annotate(row['Model'], (row['Test_MAE'], row['Test_R2']))
        axes[1, 0].set_xlabel('Test MAE (€)')
        axes[1, 0].set_ylabel('Test R²')
        axes[1, 0].set_title('Test MAE vs Test R²')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Difference between train and test
        results_df['MAE_Diff'] = results_df['Test_MAE'] - results_df['Train_MAE']
        results_df['R2_Diff'] = results_df['Test_R2'] - results_df['Train_R2']
        
        axes[1, 1].bar(results_df['Model'], results_df['MAE_Diff'], alpha=0.7, label='MAE Diff')
        axes2 = axes[1, 1].twinx()
        axes2.plot(results_df['Model'], results_df['R2_Diff'], color='red', marker='o', label='R² Diff')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('MAE Difference (€)')
        axes2.set_ylabel('R² Difference')
        axes[1, 1].set_title('Overfitting Analysis (Test - Train)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(loc='upper left')
        axes2.legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Interpretation
        st.subheader("📝 Interpretasi Hasil")
        
        best_model_idx = results_df['Test_R2'].idxmax()
        best_model = results_df.loc[best_model_idx, 'Model']
        best_r2 = results_df.loc[best_model_idx, 'Test_R2']
        best_mae = results_df.loc[best_model_idx, 'Test_MAE']
        
        st.success(f"""
        **Model Terbaik:** {best_model}
        
        - **R² Score:** {best_r2:.4f} 
          (Model menjelaskan {best_r2*100:.1f}% variasi dalam data)
        - **Mean Absolute Error:** €{best_mae:,.0f}
          (Rata-rata error prediksi)
        """)
        
        # Recommendations khusus untuk dataset Germany
        st.info("""
        **💡 Rekomendasi untuk Dataset Germany Cars:**
        
        1. **Feature engineering untuk market Jerman:**
           - Rasio HP/harga
           - Kategori merek (premium vs standard)
           - Usia mobil (tahun sekarang - tahun registrasi)
        
        2. **Handle categorical variables:**
           - Banyak merek dan model unik
           - Pertimbangkan frequency encoding
        
        3. **Market-specific factors:**
           - Pajak kendaraan Jerman
           - Biaya asuransi
           - Popularitas merek di Eropa
        """)
    
    except:
        st.warning("Hasil evaluasi untuk dataset Germany belum tersedia. Silakan training model terlebih dahulu.")

# Footer - DIUBAH untuk mencerminkan dataset baru
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🛠️ Panduan untuk Dataset Germany Cars

**Langkah-langkah:**
1. Download dataset dari: https://www.kaggle.com/datasets/ander289386/cars-germany
2. Simpan sebagai: `autoscout24-germany-dataset.csv`
3. Install dependencies: `pip install -r requirements.txt`
4. Buat script training khusus: `train_model_germany.py`
5. Run aplikasi: `streamlit run app.py`

**Fitur Dataset:**
- 46,405 kendaraan
- Tahun: 2011-2021
- Sumber: AutoScout24
- Mata uang: Euro (€)

**Kolom utama:**
- make (brand)
- model
- fuel type
- gear (transmission)
- price
- hp (horse_power)
- year (registration_year)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Umar Perdana | © 2025 | Dataset: Germany Cars")