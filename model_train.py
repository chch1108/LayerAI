import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

# --- Constants ---
DATA_FILE = 'data.csv'
MODEL_DIR = 'model_artifacts'
MODEL_PATH = os.path.join(MODEL_DIR, 'dlp_reflow_model.joblib')
TARGET_COLUMN = '回流（完全是0不完全是1）'

# --- Feature Columns ---
# Based on data.csv
CATEGORICAL_FEATURES = ['形狀']
NUMERICAL_FEATURES = [
    '材料黏度 (cps)', '抬升高度(μm)', '抬升速度(μm/s)', '等待時間(s)', 
    '下降速度((μm)/s)', '面積(mm?)', '周長(mm)', '水力直徑(mm)'
]
# Note: '荷重元最大值_Fmax', '波峰後斜率_Slope', '曲線面積_Area' seem to be results, not input parameters.
# We will only use features that are available *before* the print starts.
INPUT_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def train_and_save_model():
    """
    Loads data, trains a random forest classifier, and saves the entire
    preprocessing and model pipeline to a file.
    """
    print("Starting model training process...")

    # --- 1. Load and Prepare Data ---
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}. Please ensure it's in the same directory.")
    
    df = pd.read_csv(DATA_FILE)

    # Drop rows with missing target values if any
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    # Define features (X) and target (y)
    X = df[INPUT_FEATURES]
    y = df[TARGET_COLUMN]

    print(f"Data loaded. Shape of X: {X.shape}, Shape of y: {y.shape}")

    # --- 2. Preprocessing Pipeline ---
    # Create a preprocessor for numeric and categorical features
    # Numeric features will be scaled, categorical features will be one-hot encoded.
    numeric_transformer = StandardScaler()
    categorical_transformer = ('onehot', pd.get_dummies, CATEGORICAL_FEATURES) # Using pandas get_dummies

    # Using a simple approach for pandas get_dummies within the workflow
    X_processed = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=True)
    
    # Store the columns after one-hot encoding for consistent prediction
    processed_columns = X_processed.columns.tolist()
    
    # Update numerical features list to exclude the original categorical column
    current_numerical_features = [f for f in NUMERICAL_FEATURES if f in X_processed.columns]

    # Build a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, current_numerical_features)
        ],
        remainder='passthrough' # Keep the one-hot encoded columns as they are
    )

    # --- 3. Model Training ---
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

    # Create the full pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    # Train the model
    print("Training the model...")
    model_pipeline.fit(X_train, y_train)

    # --- 4. Evaluation ---
    print("Evaluating the model...")
    y_pred = model_pipeline.predict(X_test)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("---------------------------\n")

    # --- 5. Save Artifacts ---
    # Create directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the entire pipeline and the column list
    artifacts = {
        'pipeline': model_pipeline,
        'columns': processed_columns
    }
    joblib.dump(artifacts, MODEL_PATH)
    print(f"Model and preprocessing artifacts saved to: {MODEL_PATH}")
    
    return artifacts


def load_model_and_predict(input_data: pd.DataFrame):
    """
    Loads the saved model pipeline and makes a prediction.
    Returns probability (float 0~1) for class 1 (失敗) and feature importances.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run training first.")

    artifacts = joblib.load(MODEL_PATH)
    model_pipeline = artifacts['pipeline']
    trained_columns = artifacts['columns']

    # Preprocess input data to match training columns
    processed_input = pd.DataFrame(columns=trained_columns, index=input_data.index)
    input_encoded = pd.get_dummies(input_data, columns=CATEGORICAL_FEATURES, drop_first=True)

    for col in input_encoded.columns:
        if col in processed_input.columns:
            processed_input[col] = input_encoded[col]

    processed_input.fillna(0, inplace=True)
    processed_input = processed_input[trained_columns]

    # --- 返回失敗機率 ---
    if hasattr(model_pipeline.named_steps['classifier'], "predict_proba"):
        prob = model_pipeline.predict_proba(processed_input)[0, 1]  # class 1 的機率
    else:
        # fallback to hard prediction
        prob = float(model_pipeline.predict(processed_input)[0])

    # feature importances
    try:
        importances = model_pipeline.named_steps['classifier'].feature_importances_
        feature_importances = dict(zip(trained_columns, importances))
    except Exception:
        feature_importances = {}

    return prob, feature_importances


if __name__ == '__main__':
    # This block will run when the script is executed directly
    # It trains and saves the model.
    train_and_save_model()

    # --- Example of how to use the prediction function ---
    print("\n--- Running a test prediction ---")
    try:
        # Create a sample input (using the first row of the dataset for demonstration)
        sample_df = pd.read_csv(DATA_FILE).head(1)
        sample_input = sample_df[INPUT_FEATURES]
        
        print("Sample Input:\n", sample_input)

        prediction, importances = load_model_and_predict(sample_input)
        
        print(f"\nPrediction: {prediction} (0=OK, 1=Fail)")
        print("Feature Importances (Top 5):")
        sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)
        for feature, importance in sorted_importances[:5]:
            print(f"- {feature}: {importance:.4f}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during the test prediction: {e}")
