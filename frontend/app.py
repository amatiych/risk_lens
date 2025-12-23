"""Risk Lens - Streamlit web application for portfolio risk analysis.

This is the main entry point for the Risk Lens web application.
It provides a multi-page interface for:
- Uploading portfolio CSV files
- Viewing comprehensive risk analysis (VaR, factors, regimes)
- Chatting with Claude AI about portfolio risks

Usage:
    streamlit run frontend/app.py
    # or
    pdm run app
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.components.upload import render_upload_section, render_sample_format
from frontend.components.analysis_display import render_full_analysis
from frontend.components.chat import render_chat_interface, render_suggested_questions
from frontend.services.portfolio_service import (
    parse_uploaded_csv,
    enrich_portfolio,
    run_analysis
)
from frontend.services.chat_service import get_initial_analysis


st.set_page_config(
    page_title="Risk Lens",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize Streamlit session state variables.

    Sets up default values for portfolio, analysis results, AI summary,
    and current page navigation if they don't already exist.
    """
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'ai_summary' not in st.session_state:
        st.session_state.ai_summary = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Upload"


def main():
    """Main application entry point.

    Initializes session state, renders sidebar navigation, and routes
    to the appropriate page based on current navigation state.
    """
    init_session_state()

    st.sidebar.title("Risk Lens")
    st.sidebar.markdown("Portfolio Risk Analysis")

    if st.session_state.analysis_results is not None:
        page = st.sidebar.radio(
            "Navigation",
            ["Upload", "Analysis", "Chat"],
            index=["Upload", "Analysis", "Chat"].index(st.session_state.current_page)
        )
        st.session_state.current_page = page
    else:
        st.session_state.current_page = "Upload"
        st.sidebar.info("Upload a portfolio to begin analysis")

    if st.session_state.current_page == "Upload":
        render_upload_page()
    elif st.session_state.current_page == "Analysis":
        render_analysis_page()
    elif st.session_state.current_page == "Chat":
        render_chat_page()


def render_upload_page():
    """Render the portfolio upload page.

    Displays file upload widget, sample format, and run analysis button.
    On successful analysis, stores results in session state and navigates
    to the Analysis page.
    """
    st.title("Portfolio Upload")

    render_sample_format()

    uploaded_file, nav = render_upload_section()

    if uploaded_file is not None:
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Processing portfolio..."):
                try:
                    portfolio = parse_uploaded_csv(uploaded_file, nav)
                    st.info(f"Loaded {len(portfolio.holdings)} positions")

                    with st.spinner("Fetching market data..."):
                        portfolio = enrich_portfolio(portfolio)

                    with st.spinner("Running risk analysis..."):
                        results = run_analysis(portfolio)

                    with st.spinner("Generating AI summary..."):
                        ai_summary = get_initial_analysis(results.report)

                    st.session_state.portfolio = portfolio
                    st.session_state.analysis_results = results
                    st.session_state.ai_summary = ai_summary
                    st.session_state.current_page = "Analysis"
                    st.success("Analysis complete!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


def render_analysis_page():
    """Render the portfolio analysis results page.

    Displays tabbed interface with VaR metrics, correlations, factor exposures,
    and regime analysis. Includes sidebar button to start a new analysis.
    """
    st.title("Portfolio Analysis")

    if st.session_state.analysis_results is None:
        st.warning("No analysis results. Please upload a portfolio first.")
        return

    render_full_analysis(
        st.session_state.analysis_results,
        st.session_state.ai_summary
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("New Analysis"):
        st.session_state.portfolio = None
        st.session_state.analysis_results = None
        st.session_state.ai_summary = None
        st.session_state.chat_history = []
        st.session_state.current_page = "Upload"
        st.rerun()


def render_chat_page():
    """Render the portfolio Q&A chat page.

    Provides a chat interface for asking questions about the portfolio
    with Claude AI. Includes suggested questions sidebar.
    """
    st.title("Portfolio Q&A")

    if st.session_state.analysis_results is None:
        st.warning("No analysis results. Please upload a portfolio first.")
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        render_chat_interface(st.session_state.analysis_results.report)

    with col2:
        suggestion = render_suggested_questions()
        if suggestion:
            st.session_state.chat_history.append({
                "role": "user",
                "content": suggestion
            })
            st.rerun()


if __name__ == "__main__":
    main()
