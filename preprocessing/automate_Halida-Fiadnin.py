import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def automate_preprocessing(input_path: str, output_path: str):
    # Load data
    df = pd.read_csv(input_path)

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Handle missing values (optional fallback)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].median())

    # --- 📏 MinMax Scaling (hanya kolom numerik) ---
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # --- 🔁 Encoding ---

    # Daftar kolom dengan nilai Yes/No
    binary_cols = ['Stage_fear', 'Drained_after_socializing']
    
    # Proses encoding untuk kolom Yes/No → 0/1
    for col in binary_cols:
        df[col] = df[col].map({'No': 0, 'Yes': 1})

    # Proses LabelEncoder untuk kolom object lainnya (kecuali binary_cols)
    label_cols = df.select_dtypes(include='object').columns.difference(binary_cols)
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessing selesai. Dataset disimpan di {output_path}")

if __name__ == "__main__":
    input_path = "..\\personality_dataset_raw.csv"
    output_path = "personality_dataset_preprocessing.csv"
    automate_preprocessing(input_path, output_path)

# Trigger run for GitHub Actions