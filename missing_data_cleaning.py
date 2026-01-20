# Advanced Data Cleaning & Missing Value Handling Workflow
# Using Python (Pandas, NumPy, Matplotlib)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Section 1: Dataset Loading

print("=== Section 1: Dataset Loading ===")

# Load House Prices dataset from CSV
house_df = pd.read_csv('house_prices_train.csv')
print("Loaded House Prices dataset from CSV")

# Load Medical Appointment No Shows dataset from CSV
medical_df = pd.read_csv('noshowappointments.csv')
print("Loaded Medical Appointment No Shows dataset from CSV")

# Process both datasets
datasets = [('House Prices', house_df), ('Medical Appointment No Shows', medical_df)]

for dataset_name, original_df in datasets:
    print(f"\n{'='*50}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*50}")

    df = original_df.copy()

    print(f"Dataset: {dataset_name}")
    print(f"First 5 records:")
    print(df.head())
    print(f"\nLast 5 records:")
    print(df.tail())
    print(f"\nDataset shape: {df.shape}")
    print(f"Data types:")
    print(df.dtypes)

    # Section 2: Missing Value Detection

    print("\n=== Section 2: Missing Value Detection ===")

    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    print("Total missing values per column:")
    print(missing_values[missing_values > 0])
    print("\nPercentage of missing values per column:")
    print(missing_percentage[missing_percentage > 0])

    most_affected = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
    print(f"\nColumns most affected by missing values (sorted by percentage):")
    print(most_affected)

    # Section 3: Missing Data Visualization

    print("\n=== Section 3: Missing Data Visualization ===")

    if missing_values.sum() > 0:
        plt.figure(figsize=(12, 6))
        missing_values[missing_values > 0].plot(kind='bar', color='skyblue')
        plt.title(f'Missing Values per Column - {dataset_name}')
        plt.xlabel('Columns')
        plt.ylabel('Number of Missing Values')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No missing values to visualize.")

    # Section 4: Numerical Data Imputation

    print("\n=== Section 4: Numerical Data Imputation ===")

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Numerical columns: {list(numerical_cols)}")

    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            if abs(df[col].skew()) > 1 or abs(df[col].kurtosis()) > 3:  # Skewed or has outliers
                df[col].fillna(df[col].median(), inplace=True)
                print(f"Imputed {col} with median (skewed/outliers)")
            else:
                df[col].fillna(df[col].mean(), inplace=True)
                print(f"Imputed {col} with mean (normally distributed)")

    # Section 5: Categorical Data Imputation

    print("\n=== Section 5: Categorical Data Imputation ===")

    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"Categorical columns: {list(categorical_cols)}")

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"Imputed {col} with mode: {df[col].mode()[0]}")

    print("Mode imputation is appropriate for categorical data as it preserves the most frequent category, maintaining the distribution.")

    # Section 6: Column Removal (High Missing Rate)

    print("\n=== Section 6: Column Removal (High Missing Rate) ===")

    high_missing_cols = missing_percentage[missing_percentage > 50].index
    if len(high_missing_cols) > 0:
        print(f"Columns with >50% missing values: {list(high_missing_cols)}")
        for col in high_missing_cols:
            # For house prices, keep SalePrice; for medical, no such critical columns assumed
            critical_cols = ['SalePrice'] if 'SalePrice' in df.columns else []
            if col not in critical_cols:
                print(f"Removing column '{col}' as it has {missing_percentage[col]:.2f}% missing values and may not add analytical value.")
                df.drop(col, axis=1, inplace=True)
            else:
                print(f"Keeping column '{col}' despite high missing rate as it is critical.")
    else:
        print("No columns have >50% missing values.")

    # Section 7: Dataset Validation After Cleaning

    print("\n=== Section 7: Dataset Validation After Cleaning ===")

    missing_after = df.isnull().sum()
    print("Missing values after cleaning:")
    print(missing_after[missing_after > 0])

    if missing_after.sum() == 0:
        print("All missing values have been handled successfully.")
    else:
        print("Some missing values remain. Please review.")

    print(f"Updated dataset shape: {df.shape}")

    # Section 8: Before vs After Comparison

    print("\n=== Section 8: Before vs After Comparison ===")

    original_shape = original_df.shape
    original_missing = original_df.isnull().sum().sum()

    cleaned_shape = df.shape
    cleaned_missing = df.isnull().sum().sum()

    print(f"Original dataset shape: {original_shape}")
    print(f"Cleaned dataset shape: {cleaned_shape}")
    print(f"Original missing values count: {original_missing}")
    print(f"Cleaned missing values count: {cleaned_missing}")
    print(f"Missing values handled: {original_missing - cleaned_missing}")
    print(f"Columns removed: {original_shape[1] - cleaned_shape[1]}")

    print("\nData quality improvements:")
    print("- Missing values handled through appropriate imputation methods")
    print("- Columns with excessively high missing rates removed")
    print("- Dataset is now ready for machine learning applications")

    # Section 9: Save Cleaned Data to CSV

    print("\n=== Section 9: Save Cleaned Data to CSV ===")

    cleaned_filename = f"cleaned_{dataset_name.lower().replace(' ', '_')}.csv"
    df.to_csv(cleaned_filename, index=False)
    print(f"Cleaned dataset saved to {cleaned_filename}")

print("\n=== Overall Workflow Complete ===")
