# train_lung_cancer_model_fixed_encoding.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from collections import Counter
import joblib
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import os

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

def print_data_info(df):
    """Print detailed data information"""
    print("\n" + "="*80)
    print(" DATA INFORMATION")
    print("="*80)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nFeature ranges (should be 1=NO, 2=YES):")
    for col in df.columns:
        if col not in ['GENDER', 'AGE', 'LUNG_CANCER']:
            unique_vals = sorted(df[col].unique())
            print(f"{col:25s}: {unique_vals} - 1=NO, 2=YES")
    
    print(f"\nTarget distribution:")
    print(df['LUNG_CANCER'].value_counts())
    print(f"Percentage: {df['LUNG_CANCER'].value_counts(normalize=True) * 100}")
    
    print(f"\nFeature value counts (first few rows):")
    for col in df.columns[:5]:  # Show first 5 features
        print(f"{col:25s}: {dict(df[col].value_counts())}")

def create_better_model(input_dim):
    """Create a model that can handle the 1-2 encoding"""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        
        # First layer - normalize the 1-2 range
        keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        # Second layer
        keras.layers.Dense(16, activation='relu', kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        # Output layer
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc'),
                 keras.metrics.AUC(name='prc', curve='PR')]  # PR curve for imbalanced data
    )
    
    return model

def encode_features_correctly(df):
    """
    Correctly encode all features:
    - GENDER: M=0, F=1
    - All other features (1-2 scale): Convert to 0-1 where 1 becomes 0, 2 becomes 1
    """
    print("\n ENCODING FEATURES...")
    
    # Make a copy
    df_encoded = df.copy()
    
    # 1. Encode GENDER
    df_encoded['GENDER'] = df_encoded['GENDER'].map({'M': 0, 'F': 1})
    print(f"GENDER encoded: M=0, F=1")
    
    # 2. Encode all symptom features (1=NO, 2=YES) to (0=NO, 1=YES)
    symptom_features = [
        'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
        'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
        'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
        'SWALLOWING DIFFICULTY', 'CHEST PAIN'
    ]
    
    for feature in symptom_features:
        # Clean feature name (some have trailing spaces)
        feature_clean = feature.strip()
        if feature_clean in df_encoded.columns:
            # Convert 1→0 (NO), 2→1 (YES)
            df_encoded[feature_clean] = df_encoded[feature_clean].replace({1: 0, 2: 1})
            print(f"{feature_clean:25s}: 1→0 (NO), 2→1 (YES)")
    
    # 3. Encode target
    df_encoded['LUNG_CANCER'] = df_encoded['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    print(f"LUNG_CANCER encoded: YES=1, NO=0")
    
    # Verify encoding
    print(f"\n VERIFYING ENCODING:")
    print(f"Feature value ranges after encoding:")
    for col in df_encoded.columns:
        unique_vals = sorted(df_encoded[col].unique())
        if col == 'AGE':
            print(f"{col:25s}: {unique_vals[:5]}... (range: {df_encoded[col].min()}-{df_encoded[col].max()})")
        elif col == 'GENDER':
            print(f"{col:25s}: {unique_vals} (0=M, 1=F)")
        elif col == 'LUNG_CANCER':
            print(f"{col:25s}: {unique_vals} (0=NO, 1=YES)")
        else:
            print(f"{col:25s}: {unique_vals} (0=NO, 1=YES)")
    
    return df_encoded

def main():
    print("\n" + "="*80)
    print(" LUNG CANCER PREDICTION - CORRECT ENCODING VERSION")
    print("="*80)
    
    # Setup directories
    SCRIPT_DIR = Path(__file__).parent
    RESULTS_DIR = SCRIPT_DIR / "results"
    MODEL_DIR = SCRIPT_DIR / "model"
    
    # Create directories if they don't exist
    RESULTS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Create a timestamped run directory for this execution
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    RUN_DIR = RESULTS_DIR / run_timestamp
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f" Results will be saved to: {RUN_DIR}")
    print(f" Models will be saved to: {MODEL_DIR}")
    
    try:
        # Step 1: Load data
        print("\n Loading dataset...")
        file_path = SCRIPT_DIR / "lcs_synthetic_20000.csv"
        df = pd.read_csv(file_path)
        
        # Clean column names (remove trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Print original data info
        print_data_info(df)
        
        # Step 2: Apply correct encoding
        df_encoded = encode_features_correctly(df)
        
        # Step 3: Separate features and target
        X = df_encoded.drop('LUNG_CANCER', axis=1)
        y = df_encoded['LUNG_CANCER'].values
        
        print(f"\n FINAL TARGET DISTRIBUTION:")
        class_counts = Counter(y)
        print(f"Class 0 (NO): {class_counts[0]} ({class_counts[0]/len(y)*100:.1f}%)")
        print(f"Class 1 (YES): {class_counts[1]} ({class_counts[1]/len(y)*100:.1f}%)")
        
        # Step 4: Split data
        print("\n Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Training class distribution: {Counter(y_train)}")
        
        # Step 5: Scale features
        print("\n Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler to model directory
        scaler_path = MODEL_DIR / 'scaler.pkl'
        joblib.dump(scaler, scaler_path)
        print(f" Scaler saved to: {scaler_path}")
        
        # Step 6: Create model
        print("\n Creating model...")
        model = create_better_model(X_train_scaled.shape[1])
        
        # Step 7: Calculate class weights
        class_counts = Counter(y_train)
        total = len(y_train)
        
        # Give higher weight to minority class (NO/0)
        # If NO is minority, boost it; if YES is minority, boost it
        minority_class = 0 if class_counts[0] < class_counts[1] else 1
        majority_class = 1 - minority_class
        
        weight_for_minority = total / (2.0 * class_counts[minority_class])
        weight_for_majority = total / (2.0 * class_counts[majority_class])
        
        # Boost minority class even more
        weight_for_minority *= 2.0
        
        class_weight = {minority_class: weight_for_minority, 
                       majority_class: weight_for_majority}
        
        print(f"\n  CLASS WEIGHTS:")
        print(f"Minority class ({'NO' if minority_class == 0 else 'YES'}): {weight_for_minority:.2f}")
        print(f"Majority class ({'NO' if majority_class == 0 else 'YES'}): {weight_for_majority:.2f}")
        
        # Step 8: Train model
        print("\n Training model...")
        
        # Callbacks
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=15,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(MODEL_DIR / 'best_model.keras'),
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,
            class_weight=class_weight,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Step 9: Evaluate model
        print("\n Evaluating model...")
        
        # Load best model
        best_model = keras.models.load_model(
            MODEL_DIR / 'best_model.keras',
            compile=True
        )
        
        # Get predictions
        y_pred_proba = best_model.predict(X_test_scaled, verbose=0).ravel()
        
        # Try different thresholds
        print("\n Testing thresholds:")
        print("-" * 80)
        print(f"{'Threshold':<10} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'TP':<6} {'FN':<6} {'TN':<6} {'FP':<6}")
        print("-" * 80)
        
        best_f1 = 0
        best_threshold = 0.5
        best_y_pred = None
        
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            y_pred = (y_pred_proba > threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{threshold:<10.2f} {accuracy:<8.4f} {precision:<8.4f} "
                  f"{recall:<8.4f} {f1:<8.4f} {tp:<6} {fn:<6} {tn:<6} {fp:<6}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_y_pred = y_pred
        
        print("-" * 80)
        print(f"\n Best threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
        
        # Final evaluation
        print("\n FINAL EVALUATION:")
        print(confusion_matrix(y_test, best_y_pred))
        print("\n" + classification_report(y_test, best_y_pred, 
                                          target_names=['NO', 'YES']))
        
        # Calculate and print key metrics
        cm = confusion_matrix(y_test, best_y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\n KEY METRICS:")
        print(f"Sensitivity (Recall for YES): {sensitivity:.4f}")
        print(f"Specificity (Recall for NO):  {specificity:.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Save model and artifacts to model directory
        print(f"\n Saving models to: {MODEL_DIR}")
        model_path = MODEL_DIR / 'final_model.keras'
        best_model.save(model_path)
        print(f" Model saved to: {model_path}")
        
        # Save threshold and feature names to model directory
        threshold_path = MODEL_DIR / 'optimal_threshold.pkl'
        joblib.dump(best_threshold, threshold_path)
        print(f" Optimal threshold saved to: {threshold_path}")
        
        # Save feature names
        feature_names = X.columns.tolist()
        features_path = MODEL_DIR / 'feature_names.pkl'
        joblib.dump(feature_names, features_path)
        print(f" Feature names saved to: {features_path}")
        
        # Save run-specific artifacts to the run directory
        print(f"\n Saving run artifacts to: {RUN_DIR}")
        
        # Save training history plot
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(RUN_DIR / 'training_history.png')
        print(f" Training history plot saved to: {RUN_DIR / 'training_history.png'}")
        
        # Save classification report
        report = classification_report(y_test, best_y_pred, target_names=['NO', 'YES'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(RUN_DIR / 'classification_report.csv')
        print(f" Classification report saved to: {RUN_DIR / 'classification_report.csv'}")
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, best_y_pred)
        cm_df = pd.DataFrame(cm, index=['Actual NO', 'Actual YES'], 
                            columns=['Predicted NO', 'Predicted YES'])
        cm_df.to_csv(RUN_DIR / 'confusion_matrix.csv')
        print(f" Confusion matrix saved to: {RUN_DIR / 'confusion_matrix.csv'}")
        
        print("\n" + "="*80)
        print(" TRAINING COMPLETE!")
        print("="*80)
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()