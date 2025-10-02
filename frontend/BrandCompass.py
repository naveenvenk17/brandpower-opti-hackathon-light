#!/usr/bin/env python3
"""
BrandCompass.ai - Main Streamlit Application
A comprehensive brand power simulation platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import random
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="BrandCompass.ai",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def create_template_csv():
    """Create a template CSV for download"""
    template_data = {
        'Brand': ['BrandA', 'BrandA', 'BrandB', 'BrandB', 'BrandC', 'BrandC'],
        'Year': [2024, 2024, 2024, 2024, 2024, 2024],
        'Month': [7, 8, 7, 8, 7, 8],
        'Week': [1, 1, 1, 1, 1, 1],
        'brand_events': [100, 110, 80, 90, 120, 130],
        'brand_promotion': [200, 210, 180, 190, 220, 230],
        'digitaldisplayandsearch': [150, 160, 140, 150, 170, 180],
        'digitalvideo': [300, 310, 280, 290, 320, 330],
        'influencer': [50, 55, 45, 50, 60, 65],
        'meta': [400, 410, 380, 390, 420, 430],
        'ooh': [75, 80, 70, 75, 85, 90],
        'opentv': [500, 510, 480, 490, 520, 530],
        'others': [25, 30, 20, 25, 35, 40],
        'paytv': [200, 210, 180, 190, 220, 230],
        'radio': [100, 110, 90, 100, 120, 130],
        'sponsorship': [150, 160, 140, 150, 170, 180],
        'streamingaudio': [80, 85, 75, 80, 90, 95],
        'tiktok': [120, 125, 110, 115, 130, 135],
        'twitter': [90, 95, 85, 90, 100, 105],
        'youtube': [250, 260, 240, 250, 270, 280]
    }

    return pd.DataFrame(template_data)


def main():
    """Main application"""

    # Initialize session state
    if 'selected_country' not in st.session_state:
        st.session_state.selected_country = None
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    # Main header
    st.markdown('<h1 class="main-header">🧭 BrandCompass.ai</h1>',
                unsafe_allow_html=True)
    st.markdown("### Navigate Your Brand's Power with Precision")

    # Section: Select Country
    st.markdown('<div class="section-header">🌍 Select Country</div>',
                unsafe_allow_html=True)

    st.markdown("Choose your target market:")

    # Create 3 country buttons in a row
    col1, col2, col3 = st.columns(3)

    with col1:
        brazil_clicked = st.button(
            "🇧🇷 Brazil",
            key="brazil_btn",
            use_container_width=True,
            help="Select Brazil for brand power analysis"
        )

    with col2:
        colombia_clicked = st.button(
            "🇨🇴 Colombia",
            key="colombia_btn",
            use_container_width=True,
            help="Select Colombia for brand power analysis"
        )

    with col3:
        us_clicked = st.button(
            "🇺🇸 USA",
            key="us_btn",
            use_container_width=True,
            help="Select USA for brand power analysis"
        )

    # Handle button clicks
    if brazil_clicked:
        st.session_state.selected_country = "Brazil"
        st.rerun()
    elif colombia_clicked:
        st.session_state.selected_country = "Colombia"
        st.rerun()
    elif us_clicked:
        st.session_state.selected_country = "US"
        st.rerun()

    # Display selected country
    if st.session_state.selected_country:
        country_flags = {"Brazil": "🇧🇷", "Colombia": "🇨🇴", "US": "🇺🇸"}
        st.success(
            f"Selected: {country_flags.get(st.session_state.selected_country, '🌍')} {st.session_state.selected_country}")

    # Section: Upload Data
    st.markdown('<div class="section-header">📊 Upload Data</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your brand data CSV file:",
            type=['csv'],
            help="Upload a CSV file containing your brand marketing data"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_data = df

                st.markdown(
                    '<div class="success-message">✅ File uploaded successfully!</div>', unsafe_allow_html=True)

                # Show data preview
                with st.expander("📋 Data Preview", expanded=True):
                    st.write(
                        f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
                    st.dataframe(df.head(10), use_container_width=True)

                    # Show data summary with flexible column names
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        # Handle flexible brand column names
                        brand_col = 'Brand' if 'Brand' in df.columns else 'brand' if 'brand' in df.columns else None
                        if brand_col:
                            brands = sorted(df[brand_col].unique().tolist())
                            st.write(f"**Brands ({len(brands)}):**", ", ".join(
                                brands[:5]) + ("..." if len(brands) > 5 else ""))
                        else:
                            st.write("**Brands:**", "N/A")

                        # Handle flexible year column names
                        year_col = 'Year' if 'Year' in df.columns else 'year' if 'year' in df.columns else None
                        if year_col:
                            years = sorted(df[year_col].unique().tolist())
                            st.write(
                                f"**Years ({len(years)}):**", ", ".join(map(str, years)))
                        else:
                            st.write("**Years:**", "N/A")
                    with col_info2:
                        # Handle flexible month column names
                        month_col = 'Month' if 'Month' in df.columns else 'month' if 'month' in df.columns else None
                        if month_col:
                            months = sorted(df[month_col].unique().tolist())
                            st.write(
                                f"**Months ({len(months)}):**", ", ".join(map(str, months)))
                        else:
                            st.write("**Months:**", "N/A")

                        # Handle flexible week column names
                        week_col = 'Week' if 'Week' in df.columns else 'week' if 'week' in df.columns else 'week_of_month' if 'week_of_month' in df.columns else None
                        if week_col:
                            weeks = sorted(df[week_col].unique().tolist())
                            st.write(f"**Weeks ({len(weeks)}):**", ", ".join(
                                map(str, weeks[:10])) + ("..." if len(weeks) > 10 else ""))
                        else:
                            st.write("**Weeks:**", "N/A")

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.session_state.uploaded_data = None

    with col2:
        st.markdown("**📥 Download Template**")

        # Create template
        template_df = create_template_csv()

        # Convert to CSV
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 Download CSV Template",
            data=csv_data,
            file_name="brandcompass_template.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download a sample CSV template with the required format"
        )

        # Show template preview
        with st.expander("👀 Template Preview"):
            st.dataframe(template_df.head(3), use_container_width=True)

    # Section: Simulate
    st.markdown('<div class="section-header">🚀 Simulate</div>',
                unsafe_allow_html=True)

    # Check if ready to simulate
    can_simulate = st.session_state.selected_country and st.session_state.uploaded_data is not None

    if not can_simulate:
        missing_items = []
        if not st.session_state.selected_country:
            missing_items.append("Country selection")
        if st.session_state.uploaded_data is None:
            missing_items.append("Data upload")

        st.warning(f"Please complete: {', '.join(missing_items)}")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        simulate_button = st.button(
            "🚀 Start Simulation",
            type="primary",
            disabled=not can_simulate,
            use_container_width=True,
            help="Begin your brand power analysis" if can_simulate else "Complete the requirements above first"
        )

    if simulate_button and can_simulate:
        st.session_state.page = 'welcome'
        st.success("🎉 Simulation started! Redirecting to analysis...")
        st.balloons()

        # Add a small delay and rerun to show the welcome page
        import time
        time.sleep(1)
        st.switch_page("pages/2_Welcome.py")


if __name__ == "__main__":
    main()
