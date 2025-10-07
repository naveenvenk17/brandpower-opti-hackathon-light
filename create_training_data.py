#!/usr/bin/env python3
"""
Create training dataset by merging brand power data with marketing features

This script combines:
1. Brand power data from BG_Data_Hackathon_Train.xlsx
2. Marketing features from comprehensive_mapping_with_marketing.csv

Output: data/brand_power_with_marketing_features.csv
"""

import pandas as pd
import os

def main():
    print("=" * 80)
    print("Creating Training Dataset with Marketing Features")
    print("=" * 80)

    # Load brand power data from Excel
    print("\n1. Loading brand power data from Excel...")
    excel_path = "data/BG_Data_Hackathon_Train.xlsx"

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # Read the Excel file
    df_power = pd.read_excel(excel_path)
    print(f"   ✓ Loaded {len(df_power)} rows from Excel")
    print(f"   Columns: {list(df_power.columns)}")

    # Standardize column names to lowercase
    df_power.columns = df_power.columns.str.lower()

    # Extract required columns
    required_cols = ['year', 'quarter', 'country', 'brand', 'power']

    # Check which columns exist
    existing_cols = [col for col in required_cols if col in df_power.columns]
    missing_cols = [col for col in required_cols if col not in df_power.columns]

    if missing_cols:
        print(f"\n   ⚠ Missing columns in Excel: {missing_cols}")
        print(f"   Available columns: {list(df_power.columns)}")

        # Try alternative column names
        if 'original_power' in df_power.columns and 'power' not in df_power.columns:
            df_power['power'] = df_power['original_power']
            print("   ✓ Mapped 'original_power' -> 'power'")

        if 'original_country' in df_power.columns and 'country' not in df_power.columns:
            df_power['country'] = df_power['original_country']
            print("   ✓ Mapped 'original_country' -> 'country'")

        if 'original_brand' in df_power.columns and 'brand' not in df_power.columns:
            df_power['brand'] = df_power['original_brand']
            print("   ✓ Mapped 'original_brand' -> 'brand'")

    # Select only the required columns
    df_power = df_power[required_cols].copy()
    print(f"\n   Selected columns: {list(df_power.columns)}")
    print(f"   Sample data:")
    print(df_power.head(3))

    # Load marketing features
    print("\n2. Loading marketing features from CSV...")
    csv_path = "comprehensive_mapping_with_marketing.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df_marketing = pd.read_csv(csv_path)
    print(f"   ✓ Loaded {len(df_marketing)} rows from CSV")
    print(f"   Columns: {list(df_marketing.columns)}")

    # Standardize column names
    df_marketing.columns = df_marketing.columns.str.lower()

    # Handle alternative column names in marketing CSV FIRST
    if 'original_country' in df_marketing.columns:
        df_marketing['country'] = df_marketing['original_country']
        print("   ✓ Mapped 'original_country' -> 'country'")
    if 'original_brand' in df_marketing.columns:
        df_marketing['brand'] = df_marketing['original_brand']
        print("   ✓ Mapped 'original_brand' -> 'brand'")

    # Marketing feature columns (excluding ID columns and power)
    marketing_cols = [col for col in df_marketing.columns
                     if col not in ['year', 'quarter', 'country', 'brand', 'power',
                                   'original_country', 'original_brand', 'original_power']]

    print(f"\n   Marketing feature columns: {marketing_cols}")

    # Select ID columns + marketing features
    merge_cols = ['year', 'quarter', 'country', 'brand'] + marketing_cols
    df_marketing = df_marketing[merge_cols].copy()

    # Merge datasets
    print("\n3. Merging datasets...")
    print(f"   Brand power data: {len(df_power)} rows")
    print(f"   Marketing data: {len(df_marketing)} rows")

    # Merge on year, quarter, country, brand
    df_merged = pd.merge(
        df_power,
        df_marketing,
        on=['year', 'quarter', 'country', 'brand'],
        how='left'
    )

    print(f"\n   ✓ Merged result: {len(df_merged)} rows")

    # Fill missing marketing features with 0
    for col in marketing_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)

    # Verify no duplicates in key columns
    duplicates = df_merged.duplicated(subset=['year', 'quarter', 'country', 'brand'], keep=False)
    if duplicates.any():
        print(f"\n   ⚠ Warning: Found {duplicates.sum()} duplicate rows")
        print(df_merged[duplicates][['year', 'quarter', 'country', 'brand']].head())

    # Show summary statistics
    print("\n4. Dataset Summary:")
    print(f"   Total rows: {len(df_merged)}")
    print(f"   Total columns: {len(df_merged.columns)}")
    print(f"   Countries: {df_merged['country'].nunique()}")
    print(f"   Brands: {df_merged['brand'].nunique()}")
    print(f"   Date range: {df_merged['year'].min()}-{df_merged['year'].max()}")
    print(f"   Quarters: {sorted(df_merged['quarter'].unique())}")

    print("\n   Sample merged data:")
    print(df_merged.head(3))

    print("\n   Column types:")
    for col in df_merged.columns:
        print(f"   - {col}: {df_merged[col].dtype}")

    # Save to CSV
    output_path = "data/brand_power_with_marketing_features.csv"
    os.makedirs("data", exist_ok=True)

    print(f"\n5. Saving to {output_path}...")
    df_merged.to_csv(output_path, index=False)
    print(f"   ✓ Saved {len(df_merged)} rows to {output_path}")

    # Verify file was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"   ✓ File size: {file_size:.2f} KB")

    print("\n" + "=" * 80)
    print("Training dataset created successfully!")
    print("=" * 80)

    return df_merged


if __name__ == "__main__":
    try:
        df = main()
        print("\n✓ Success! Training data is ready.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
