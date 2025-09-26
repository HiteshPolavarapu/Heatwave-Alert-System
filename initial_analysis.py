"""
Complete Data Analysis and Machine Learning Pipeline for the
Summer Heat Waves Mobile Alert System - Vijayawada Prototype.

This script performs the following steps:
1.  Loads the historical weather data.
2.  Cleans and preprocesses the data (selects columns, converts units).
3.  Performs exploratory data analysis with visualizations.
4.  Engineers the 'alert_level' target feature.
5.  Prepares the data for machine learning (lag features, train-test split).
6.  Trains and evaluates an initial RandomForestClassifier model (V1).
7.  Performs hyperparameter tuning using GridSearchCV to find the best model settings.
8.  Evaluates the final, tuned model.
9.  Saves the best-performing model and its corresponding label encoder to disk.
"""

# --- SECTION 0: IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import joblib
import warnings

warnings.filterwarnings('ignore')

def main():
    """Main function to run the entire pipeline."""
    
    # --- SECTION 1: DATA LOADING ---
    print("--- Starting Phase 1: Data Loading and Cleaning ---")
    file_name = "Vijayawada24_25.csv"
    try:
        df = pd.read_csv(file_name)
        print(f"✅ Successfully loaded '{file_name}'!")
    except FileNotFoundError:
        print(f"--- ERROR: Could not find '{file_name}'. Please ensure it is in the correct directory. ---")
        return

    # --- SECTION 2: DATA CLEANING AND PREPARATION ---
    columns_to_keep = [
        'datetime', 'tempmax', 'tempmin', 'temp', 
        'feelslikemax', 'feelslike', 'humidity', 'solarradiation', 'uvindex'
    ]
    df_cleaned = df[columns_to_keep].copy()
    df_cleaned['datetime'] = pd.to_datetime(df_cleaned['datetime'])
    for col in ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslike']:
        df_cleaned[col] = (df_cleaned[col] - 32) * 5 / 9
    df_cleaned.rename(columns={
        'tempmax': 'tempmax_c', 'tempmin': 'tempmin_c', 'temp': 'temp_c',
        'feelslikemax': 'feelslikemax_c', 'feelslike': 'feelslike_c'
    }, inplace=True)
    print("✅ Data cleaning and conversion to Celsius complete!")
    print("\nPreview of cleaned data:")
    print(df_cleaned.head())

    # --- SECTION 3: EXPLORATORY DATA ANALYSIS (PLOTTING) ---
    print("\n📈 Generating Plot 1: Maximum Temperatures Over the Year...")
    plt.figure(figsize=(18, 8))
    plt.plot(df_cleaned['datetime'], df_cleaned['tempmax_c'], label='Maximum Temperature (°C)', color='red')
    plt.axhline(y=40, color='r', linestyle='--', label='IMD Heatwave Threshold (40°C)')
    plt.axhline(y=45, color='darkred', linestyle=':', label='IMD Severe Heatwave Threshold (45°C)')
    plt.title('Maximum Daily Temperature (2024-2025)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show() # Script will pause here until you close the plot window

    print("\n📈 Generating Plot 2: Actual vs. 'Feels Like' Temperature...")
    summer_df = df_cleaned[df_cleaned['datetime'].dt.month.isin([3, 4, 5, 6, 7])].copy()
    plt.figure(figsize=(18, 9))
    plt.plot(summer_df['datetime'], summer_df['tempmax_c'], label='Actual Max Temperature (°C)', color='orange', linewidth=2)
    plt.plot(summer_df['datetime'], summer_df['feelslikemax_c'], label='"Feels Like" Max Temperature (°C)', color='red', linestyle='-.', linewidth=2)
    plt.axhline(y=40, color='grey', linestyle='--', label='IMD Heatwave Threshold (40°C)')
    plt.fill_between(summer_df['datetime'], summer_df['tempmax_c'], summer_df['feelslikemax_c'], where=(summer_df['feelslikemax_c'] > summer_df['tempmax_c']), color='red', alpha=0.15, interpolate=True, label='Danger Zone (Humidity Impact)')
    plt.title('Actual vs. "Feels Like" Temperature During Hot Season', fontsize=18)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show() # Script will pause here

    # --- SECTION 4: FEATURE ENGINEERING ---
    print("\n--- Starting Phase 2: Feature Engineering and Model Development ---")
    conditions = [
        (df_cleaned['feelslikemax_c'] >= 45),
        (df_cleaned['feelslikemax_c'] >= 40) & (df_cleaned['feelslikemax_c'] < 45),
        (df_cleaned['feelslikemax_c'] < 40)
    ]
    alert_levels = ['Severe Heatwave', 'Heatwave', 'Normal']
    df_cleaned['alert_level'] = np.select(conditions, alert_levels, default='Normal')
    print("✅ New 'alert_level' column created.")
    alert_counts = df_cleaned['alert_level'].value_counts()
    print("\n--- Heatwave Risk Analysis ---")
    print(alert_counts)
    
    print("\n📊 Generating Plot 3: Bar Chart of Alert Days...")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='alert_level', data=df_cleaned, palette={'Normal': 'green', 'Heatwave': 'orange', 'Severe Heatwave': 'red'}, order=['Normal', 'Heatwave', 'Severe Heatwave'])
    plt.title('Number of Days per Alert Level', fontsize=16)
    plt.xlabel('Alert Level', fontsize=12)
    plt.ylabel('Number of Days', fontsize=12)
    plt.show() # Script will pause here

    # --- SECTION 5: DATA PREPARATION FOR ML ---
    df_model = df_cleaned.copy()
    df_model['tempmax_c_lag1'] = df_model['tempmax_c'].shift(1)
    df_model['feelslikemax_c_lag1'] = df_model['feelslikemax_c'].shift(1)
    df_model['humidity_lag1'] = df_model['humidity'].shift(1)
    df_model['month'] = df_model['datetime'].dt.month
    df_model['day_of_year'] = df_model['datetime'].dt.dayofyear
    df_model = df_model.dropna()
    features = ['tempmax_c_lag1', 'feelslikemax_c_lag1', 'humidity_lag1', 'month', 'day_of_year']
    X = df_model[features]
    y = df_model['alert_level']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    print("\n✅ Data successfully split into training and testing sets.")

    # --- SECTION 6: HYPERPARAMETER TUNING (IMPROVING V1 MODEL) ---
    print("\n🔧 Tuning the V1 model to find its best settings... (This may take a minute)")
    param_grid = {
        'n_estimators': [100, 200], 'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2], 'class_weight': ['balanced', None]
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_rf_model = grid_search.best_estimator_
    print("\n✅ Tuning complete!")
    print(f"🏆 Best settings found: {grid_search.best_params_}")

    # --- SECTION 7: EVALUATE FINAL TUNED MODEL ---
    print("\n📈 Evaluating the final, tuned model...")
    y_pred_tuned = best_rf_model.predict(X_test)
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    print(f"\n🚀 Final Tuned Model Accuracy: {accuracy_tuned * 100:.2f}%")
    print("\n--- Final Tuned Classification Report ---")
    print(classification_report(y_test, y_pred_tuned, target_names=le.classes_))
    
    print("\n📊 Generating Final Confusion Matrix...")
    ConfusionMatrixDisplay.from_estimator(best_rf_model, X_test, y_test, display_labels=le.classes_, cmap='viridis', xticks_rotation='vertical')
    plt.title('Confusion Matrix (Final Tuned Model)')
    plt.show() # Script will pause here

    # --- SECTION 8: SAVING THE PRODUCTION MODEL ---
    print("\n--- Starting Phase 3: Packaging Model for Deployment ---")
    model_filename = 'heatwave_alert_model_final.joblib'
    encoder_filename = 'alert_level_encoder_final.joblib'
    joblib.dump(best_rf_model, model_filename)
    joblib.dump(le, encoder_filename)
    print(f"\n✅ Success! Final model saved to '{model_filename}'")
    print(f"✅ Final encoder saved to '{encoder_filename}'")
    print("\n--- PROJECT COMPLETE ---")

if __name__ == '__main__':
    main()
