import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from utils import *

# Configure Streamlit page
st.set_page_config(
    page_title="BrandCompass.ai",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'selected_country' not in st.session_state:
    st.session_state.selected_country = None
if 'baseline_data' not in st.session_state:
    st.session_state.baseline_data = None


def load_baseline_data():
    """Load baseline forecast data"""
    baseline_path = os.path.join("data", "baseline_forecast.csv")
    try:
        baseline_data = pd.read_csv(baseline_path)
        return baseline_data
    except:
        return None


def landing_page():
    """Landing Page - Page 1"""
    st.title("🧭 Welcome to BrandCompass.ai")
    st.subheader("Simulate Power for")

    # Country selection buttons
    st.markdown("### Select Country")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🇧🇷 Brazil", use_container_width=True, type="primary" if st.session_state.selected_country == "Brazil" else "secondary"):
            st.session_state.selected_country = "Brazil"

    with col2:
        if st.button("🇺🇸 USA", use_container_width=True, type="primary" if st.session_state.selected_country == "USA" else "secondary"):
            st.session_state.selected_country = "USA"

    with col3:
        if st.button("🇨🇴 Colombia", use_container_width=True, type="primary" if st.session_state.selected_country == "Colombia" else "secondary"):
            st.session_state.selected_country = "Colombia"

    if st.session_state.selected_country:
        st.success(f"Selected country: {st.session_state.selected_country}")

    st.markdown("---")

    # File upload section
    st.markdown("### Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your CSV file with brand and marketing data"
    )

    if uploaded_file is not None:
        try:
            st.session_state.uploaded_data = pd.read_csv(uploaded_file)
            st.success(
                f"✅ File uploaded successfully! ({len(st.session_state.uploaded_data)} rows)")

            # Show preview
            with st.expander("Preview uploaded data"):
                st.dataframe(st.session_state.uploaded_data.head())
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    # Template download
    st.markdown("### Download Template")
    template_path = os.path.join("data", "template.csv")

    try:
        if os.path.exists(template_path):
            with open(template_path, "rb") as file:
                st.download_button(
                    label="📥 Download CSV Template",
                    data=file,
                    file_name="template.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.warning("Template file not found at expected location")
            # Show alternative download option
            st.info("You can use the test data file instead:")
            test_path = os.path.join("data", "test_quarterly_data.csv")
            if os.path.exists(test_path):
                with open(test_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Test Data",
                        data=file,
                        file_name="test_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    except Exception as e:
        st.error(f"Error accessing template file: {str(e)}")

    st.markdown("---")

    # Simulate button
    if st.button("🚀 Simulate", use_container_width=True, type="primary"):
        if st.session_state.uploaded_data is not None:
            st.session_state.page = 'simulation'
            st.rerun()
        else:
            st.error("Please upload a CSV file first!")


def simulation_page():
    """Simulation Page - Page 2"""
    st.title("📊 Simulation Dashboard")

    # Back button
    if st.button("← Back to Landing", type="secondary"):
        st.session_state.page = 'landing'
        st.rerun()

    if st.session_state.uploaded_data is None:
        st.error("No data uploaded. Please go back and upload a CSV file.")
        return

    data = st.session_state.uploaded_data

    # Load baseline data
    if st.session_state.baseline_data is None:
        st.session_state.baseline_data = load_baseline_data()

    # Sidebar for filters
    st.sidebar.header("🎛️ Simulation Filters")

    # Brand selection
    brands = get_brands_from_data(data)
    if brands:
        selected_brand = st.sidebar.selectbox("Select Brand", brands)

        # Filter data for selected brand
        brand_data = data[data['brand'] ==
                          selected_brand] if 'brand' in data.columns else data
    else:
        st.sidebar.warning("No brands found in data")
        brand_data = data
        selected_brand = None

    # Get optimizable columns
    optimizable_cols = get_optimizable_columns()

    # Create filters for optimizable columns
    st.sidebar.subheader("Adjust Marketing Spend")
    filters = {}

    # Roll weekly data to quarterly first
    try:
        quarterly_data = roll_data_to_quarter(brand_data)
    except Exception as e:
        st.error(f"Error converting to quarterly data: {str(e)}")
        quarterly_data = None

    # Filter data after Q2 2024 (quarterly level)
    if quarterly_data is not None and not quarterly_data.empty:
        # Filter for specific quarters: 2024 Q3, Q4, 2025 Q1, Q2
        target_quarters = [
            (2024, 'Q3'), (2024, 'Q4'),
            (2025, 'Q1'), (2025, 'Q2')
        ]

        try:
            if 'year' in quarterly_data.columns and 'quarter' in quarterly_data.columns:
                quarterly_mask = quarterly_data.apply(
                    lambda row: (row['year'], row['quarter']) in target_quarters, axis=1
                )
                future_quarterly_data = quarterly_data[quarterly_mask]

                if future_quarterly_data.empty:
                    st.warning(
                        "No data found for target quarters (2024 Q3, Q4, 2025 Q1, Q2)")
                    future_quarterly_data = quarterly_data  # Use all available data as fallback
            else:
                st.warning("Year or quarter columns not found in data")
                future_quarterly_data = quarterly_data
        except Exception as e:
            st.error(f"Error filtering quarterly data: {str(e)}")
            future_quarterly_data = quarterly_data

        # Display allowed features for simulation
        st.sidebar.subheader("📋 Allowed Features for Simulation")
        with st.sidebar.expander("View All Features"):
            for i, feature in enumerate(optimizable_cols, 1):
                st.sidebar.write(f"{i}. {feature.replace('_', ' ').title()}")

        # Create filters for optimizable columns
        for col in optimizable_cols:
            if col in future_quarterly_data.columns:
                current_value = future_quarterly_data[col].mean()
                if pd.isna(current_value) or current_value == 0 or np.isnan(current_value):
                    current_value = 100  # Default value

                min_val = max(0, current_value * 0.5)
                max_val = current_value * 2

                # Calculate step, ensuring it's not zero
                step_val = max(0.01, float(max_val * 0.01))
                if step_val == 0 or pd.isna(step_val) or np.isnan(step_val):
                    step_val = 1.0

                filters[col] = st.sidebar.slider(
                    col.replace('_', ' ').title(),
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(current_value),
                    step=step_val
                )
    else:
        future_quarterly_data = None

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Power Forecast")

        # Baseline forecast chart
        if st.session_state.baseline_data is not None:
            baseline_quarterly = roll_data_to_quarter(
                st.session_state.baseline_data)

            if baseline_quarterly is not None and selected_brand:
                # Filter baseline for selected brand if possible
                if 'Brand' in baseline_quarterly.columns:
                    brand_baseline = baseline_quarterly[baseline_quarterly['Brand']
                                                        == selected_brand]
                else:
                    brand_baseline = baseline_quarterly

                # Create the plot
                fig = make_subplots(
                    rows=1, cols=1,
                    subplot_titles=["Baseline vs Simulated Power Forecast"]
                )

                # Add baseline line
                if not brand_baseline.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=brand_baseline['Quarter'] if 'Quarter' in brand_baseline.columns else range(
                                len(brand_baseline)),
                            y=brand_baseline['Predicted Power'] if 'Predicted Power' in brand_baseline.columns else brand_baseline.iloc[:, -1],
                            mode='lines+markers',
                            name='Baseline Forecast',
                            line=dict(color='blue', width=3)
                        )
                    )

                # Simulate button and new forecast
                if st.button("🎯 Simulate Now", type="primary"):
                    try:
                        # Generate simulated data using filters
                        simulated_data = forecast_power(
                            future_quarterly_data, filters)

                        if simulated_data is not None and not simulated_data.empty:
                            simulated_quarterly = roll_data_to_quarter(
                                simulated_data)

                            if simulated_quarterly is not None and not simulated_quarterly.empty:
                                # Add simulated line
                                fig.add_trace(
                                    go.Scatter(
                                        x=simulated_quarterly['quarter'] if 'quarter' in simulated_quarterly.columns else range(
                                            len(simulated_quarterly)),
                                        y=simulated_quarterly['power'] if 'power' in simulated_quarterly.columns else simulated_quarterly.iloc[:, -1],
                                        mode='lines+markers',
                                        name='Simulated Forecast',
                                        line=dict(
                                            color='red', width=3, dash='dash')
                                    )
                                )

                                st.success("✅ Simulation completed!")
                            else:
                                st.error(
                                    "Error: Simulated quarterly data is empty")
                        else:
                            st.error(
                                "Error: Could not generate simulated data")
                    except Exception as e:
                        st.error(f"Error during simulation: {str(e)}")

                fig.update_layout(
                    title="Power Forecast Comparison",
                    xaxis_title="Quarter",
                    yaxis_title="Power",
                    height=500,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select a brand to view forecast")
        else:
            st.warning("Baseline data not available")

        # Display quarterly power data for all brands
        if future_quarterly_data is not None and not future_quarterly_data.empty:
            st.subheader("📊 Quarterly Power Data by Brand")
            st.write("**Power values for 2024 Q3, Q4, 2025 Q1, Q2:**")

            if 'brand' in future_quarterly_data.columns and 'power' in future_quarterly_data.columns:
                # Create a summary table
                brands_summary = {}
                for brand in sorted(future_quarterly_data['brand'].unique()):
                    brand_data = future_quarterly_data[future_quarterly_data['brand'] == brand]
                    brands_summary[brand] = {}
                    for _, row in brand_data.iterrows():
                        quarter_key = f"{row['year']} {row['quarter']}"
                        brands_summary[brand][quarter_key] = round(
                            row['power'], 2) if pd.notna(row['power']) and not np.isnan(row['power']) else 'N/A'

                # Convert to DataFrame for better display
                if brands_summary:
                    summary_df = pd.DataFrame(brands_summary).T
                    # Ensure columns are in the right order
                    desired_order = ['2024 Q3',
                                     '2024 Q4', '2025 Q1', '2025 Q2']
                    available_cols = [
                        col for col in desired_order if col in summary_df.columns]
                    if available_cols:
                        summary_df = summary_df[available_cols]

                    st.dataframe(summary_df, use_container_width=True)

                    # Also print individual brand data
                    with st.expander("📋 Detailed Brand Power Data"):
                        for brand in sorted(future_quarterly_data['brand'].unique()):
                            st.write(f"**{brand}:**")
                            brand_data = future_quarterly_data[future_quarterly_data['brand'] == brand]
                            for _, row in brand_data.iterrows():
                                quarter_label = f"{row['year']} {row['quarter']}"
                                power_value = row['power']
                                if pd.notna(power_value) and not np.isnan(power_value):
                                    st.write(
                                        f"  • {quarter_label}: {power_value:.2f}")
                                else:
                                    st.write(f"  • {quarter_label}: N/A")
            else:
                st.info("Power data not available in the uploaded dataset")

    with col2:
        st.subheader("📋 Current Settings")

        if selected_brand:
            st.write(f"**Brand:** {selected_brand}")

        if st.session_state.selected_country:
            st.write(f"**Country:** {st.session_state.selected_country}")

        st.write(f"**Data Points:** {len(data)}")

        if filters:
            st.subheader("🎛️ Active Filters")
            for col, value in filters.items():
                st.write(f"**{col.replace('_', ' ').title()}:** {value:.2f}")

# Main app logic


def main():
    if st.session_state.page == 'landing':
        landing_page()
    elif st.session_state.page == 'simulation':
        simulation_page()


if __name__ == "__main__":
    main()
