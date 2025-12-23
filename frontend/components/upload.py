"""Portfolio upload UI components for the Risk Lens application.

This module provides Streamlit components for uploading portfolio CSV files,
previewing data, and downloading sample templates.
"""

import streamlit as st
import pandas as pd


def render_upload_section():
    """Render the portfolio upload section with file uploader and NAV input.

    Displays instructions, file uploader widget, optional NAV input,
    and a preview of the uploaded data.

    Returns:
        Tuple of (uploaded_file, nav) where uploaded_file is the Streamlit
        UploadedFile object (or None) and nav is the optional NAV value.
    """
    st.header("Upload Portfolio")

    st.markdown("""
    Upload a CSV file with your portfolio holdings.

    **Required columns:**
    - `ticker` - Stock symbol (e.g., AAPL, MSFT)
    - `shares` - Number of shares
    """)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV with 'ticker' and 'shares' columns"
    )

    nav = st.number_input(
        "Portfolio NAV (optional)",
        min_value=0.0,
        value=0.0,
        help="Leave at 0 to calculate from market values"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)

            if 'ticker' not in df.columns or 'shares' not in df.columns:
                st.error("CSV must contain 'ticker' and 'shares' columns")
                return None, None

            st.subheader("Preview")
            st.dataframe(df[['ticker', 'shares']], use_container_width=True)

            return uploaded_file, nav if nav > 0 else None

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None, None

    return None, None


def render_sample_format():
    """Show sample CSV format in an expandable section.

    Displays an example portfolio DataFrame and provides a download
    button for a sample CSV file that users can use as a template.
    """
    with st.expander("Sample CSV Format"):
        sample_data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            'shares': [100, 50, 25, 30, 40]
        })
        st.dataframe(sample_data, use_container_width=True)

        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="Download Sample CSV",
            data=csv,
            file_name="sample_portfolio.csv",
            mime="text/csv"
        )
