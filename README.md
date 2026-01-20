# MLTASK2: Advanced Data Cleaning & Missing Value Handling

This project provides a comprehensive Python script for handling missing data in datasets, demonstrating best practices for data preprocessing in machine learning workflows.

## Overview

The `missing_data_cleaning.py` script implements an advanced data cleaning pipeline that:

- Loads datasets from CSV files
- Detects and analyzes missing values
- Applies appropriate imputation methods (mean/median for numerical, mode for categorical)
- Removes columns with excessively high missing rates (>50%)
- Validates cleaning results
- Saves cleaned datasets to new CSV files

## Datasets

The script processes two datasets:

1. **House Prices Dataset** (`house_prices_train.csv`) - Real estate data with various property features
2. **Medical Appointment No Shows Dataset** (`noshowappointments.csv`) - Healthcare appointment data

## Features

- **Missing Value Detection**: Identifies and quantifies missing values per column
- **Data Visualization**: Plots missing value distributions (when applicable)
- **Numerical Imputation**: Uses mean for normally distributed data, median for skewed/outlier-prone data
- **Categorical Imputation**: Uses mode (most frequent value) to preserve distribution
- **Column Removal**: Automatically removes columns with >50% missing values
- **Data Validation**: Ensures all missing values are handled
- **CSV Integration**: Loads from and saves cleaned data to CSV files

## Dependencies

- pandas
- numpy
- matplotlib

## Usage

1. Ensure the CSV datasets (`house_prices_train.csv` and `noshowappointments.csv`) are in the same directory as the script.
2. Run the script:
   ```bash
   python missing_data_cleaning.py
   ```
3. The script will output cleaning progress and save cleaned datasets as:
   - `cleaned_house_prices.csv`
   - `cleaned_medical_appointment_no_shows.csv`

## Output

The script provides detailed console output including:

- Dataset loading confirmation
- Missing value statistics
- Imputation decisions and results
- Column removal rationale
- Before/after comparisons
- File save confirmations

## Data Quality Improvements

After running the cleaning pipeline, datasets are ready for machine learning applications with:

- No missing values
- Appropriate imputation preserving data distributions
- Removal of non-informative columns
- Maintained data integrity

## Project Structure

- `missing_data_cleaning.py` - Main cleaning script
- `house_prices_train.csv` - Input house prices dataset
- `noshowappointments.csv` - Input medical appointments dataset
- `cleaned_house_prices.csv` - Output cleaned house prices (generated)
- `cleaned_medical_appointment_no_shows.csv` - Output cleaned medical data (generated)
- `README.md` - This documentation
