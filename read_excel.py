import pandas as pd

# Read the Excel file
try:
    df = pd.read_excel('data.xlsx')
    print("Excel file loaded successfully!")
    print("\nBasic info about the data:")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nSummary statistics:")
    print(df.describe())
except Exception as e:
    print(f"Error reading Excel file: {e}")