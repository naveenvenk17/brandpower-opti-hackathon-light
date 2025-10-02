#!/usr/bin/env python3
"""
Demo data generator for BrandCompass.ai
Creates sample datasets for testing the application
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


def generate_demo_data(num_brands=3, num_weeks=8):
    """Generate comprehensive demo data for BrandCompass.ai"""

    # Brand names
    # BrandA, BrandB, BrandC
    brands = [f"Brand{chr(65+i)}" for i in range(num_brands)]

    # Time periods (2024 Q3 and Q4)
    years = [2024]
    months = [7, 8, 9, 10, 11, 12]  # July to December
    weeks = list(range(1, 5))  # Weeks 1-4 in each month

    # Marketing channels (optimizable features)
    marketing_channels = [
        'brand_events', 'brand_promotion', 'digitaldisplayandsearch',
        'digitalvideo', 'influencer', 'meta', 'ooh', 'opentv',
        'others', 'paytv', 'radio', 'sponsorship', 'streamingaudio',
        'tiktok', 'twitter', 'youtube'
    ]

    # Generate data
    data = []

    for brand in brands:
        # Each brand has different base spending levels
        brand_multiplier = random.uniform(0.8, 1.2)

        for year in years:
            for month in months:
                for week in weeks:
                    row = {
                        'Brand': brand,
                        'Year': year,
                        'Month': month,
                        'Week': week
                    }

                    # Generate spending for each channel
                    for channel in marketing_channels:
                        # Base spending varies by channel type
                        if channel in ['opentv', 'meta', 'digitalvideo']:
                            base_spend = random.uniform(
                                300, 600)  # High spend channels
                        elif channel in ['youtube', 'digitaldisplayandsearch', 'brand_promotion']:
                            base_spend = random.uniform(
                                150, 300)  # Medium spend channels
                        else:
                            base_spend = random.uniform(
                                50, 150)   # Lower spend channels

                        # Apply brand multiplier and some randomness
                        spend = base_spend * brand_multiplier * \
                            random.uniform(0.7, 1.3)
                        row[channel] = round(spend, 2)

                    data.append(row)

    return pd.DataFrame(data)


def generate_extended_demo_data():
    """Generate extended demo data with more brands and time periods"""

    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']

    # Extended time period (2024 Q2 to 2025 Q2)
    time_periods = [
        (2024, 4), (2024, 5), (2024, 6),  # Q2 2024
        (2024, 7), (2024, 8), (2024, 9),  # Q3 2024
        (2024, 10), (2024, 11), (2024, 12),  # Q4 2024
        (2025, 1), (2025, 2), (2025, 3),  # Q1 2025
        (2025, 4), (2025, 5), (2025, 6),  # Q2 2025
    ]

    marketing_channels = [
        'brand_events', 'brand_promotion', 'digitaldisplayandsearch',
        'digitalvideo', 'influencer', 'meta', 'ooh', 'opentv',
        'others', 'paytv', 'radio', 'sponsorship', 'streamingaudio',
        'tiktok', 'twitter', 'youtube'
    ]

    data = []

    for brand in brands:
        # Brand-specific characteristics
        brand_profiles = {
            'BrandA': {'multiplier': 1.2, 'digital_focus': True},
            'BrandB': {'multiplier': 0.9, 'digital_focus': False},
            'BrandC': {'multiplier': 1.1, 'digital_focus': True},
            'BrandD': {'multiplier': 0.8, 'digital_focus': False},
            'BrandE': {'multiplier': 1.3, 'digital_focus': True},
        }

        profile = brand_profiles.get(
            brand, {'multiplier': 1.0, 'digital_focus': True})

        for year, month in time_periods:
            for week in range(1, 5):  # 4 weeks per month
                row = {
                    'Brand': brand,
                    'Year': year,
                    'Month': month,
                    'Week': week
                }

                # Generate spending based on brand profile
                for channel in marketing_channels:
                    # Digital vs traditional channel preferences
                    digital_channels = ['digitaldisplayandsearch', 'digitalvideo', 'meta',
                                        'tiktok', 'twitter', 'youtube', 'streamingaudio']

                    if channel in digital_channels and profile['digital_focus']:
                        base_spend = random.uniform(200, 500)
                    elif channel not in digital_channels and not profile['digital_focus']:
                        base_spend = random.uniform(200, 400)
                    else:
                        base_spend = random.uniform(50, 200)

                    # Apply seasonal trends (higher spending in Q4)
                    seasonal_factor = 1.3 if month in [11, 12] else 1.0

                    # Apply brand multiplier
                    spend = base_spend * \
                        profile['multiplier'] * seasonal_factor

                    # Add some randomness
                    spend *= random.uniform(0.8, 1.2)

                    row[channel] = round(spend, 2)

                data.append(row)

    return pd.DataFrame(data)


def save_demo_files():
    """Save demo data files"""

    # Generate basic demo data
    basic_demo = generate_demo_data()
    basic_demo.to_csv('data/demo_basic.csv', index=False)
    print(f"✅ Generated basic demo data: {len(basic_demo)} rows")

    # Generate extended demo data
    extended_demo = generate_extended_demo_data()
    extended_demo.to_csv('data/demo_extended.csv', index=False)
    print(f"✅ Generated extended demo data: {len(extended_demo)} rows")

    # Generate country-specific demo data
    countries = ['Brazil', 'Colombia', 'US']

    for country in countries:
        country_demo = generate_demo_data(num_brands=4)
        country_demo['Country'] = country

        # Adjust spending levels by country
        country_multipliers = {'Brazil': 0.8, 'Colombia': 0.7, 'US': 1.2}
        multiplier = country_multipliers.get(country, 1.0)

        marketing_channels = [col for col in country_demo.columns
                              if col not in ['Brand', 'Year', 'Month', 'Week', 'Country']]

        for channel in marketing_channels:
            country_demo[channel] = country_demo[channel] * multiplier

        filename = f'data/demo_{country.lower()}.csv'
        country_demo.to_csv(filename, index=False)
        print(f"✅ Generated {country} demo data: {len(country_demo)} rows")

    print("\n🎉 All demo files generated successfully!")
    print("\n📁 Generated files:")
    print("  - data/demo_basic.csv (basic 3-brand dataset)")
    print("  - data/demo_extended.csv (5-brand extended timeline)")
    print("  - data/demo_brazil.csv (Brazil-specific data)")
    print("  - data/demo_colombia.csv (Colombia-specific data)")
    print("  - data/demo_us.csv (US-specific data)")


if __name__ == "__main__":
    print("🚀 Generating demo data for BrandCompass.ai...\n")
    save_demo_files()
