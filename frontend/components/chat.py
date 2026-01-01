import streamlit as st
from typing import List, Dict

from backend.reporting.portfolio_report import PortfolioReport
from frontend.services.chat_service import stream_chat_response


def init_chat_state():
    """Initialize chat session state."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def render_chat_messages():
    """Render chat message history."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_chat_interface(report: PortfolioReport):
    """Render the chat interface."""
    init_chat_state()

    st.header("Ask Questions About Your Portfolio")

    st.markdown("""
    Ask questions about your portfolio's risk profile, positions,
    factor exposures, or any other aspects of the analysis.
    """)

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    render_chat_messages()

    if prompt := st.chat_input("Ask a question about your portfolio..."):
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history
            ]

            for chunk in stream_chat_response(messages, report):
                full_response += chunk
                response_placeholder.markdown(full_response + "...")

            response_placeholder.markdown(full_response)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response
        })


def render_suggested_questions():
    """Render suggested questions."""
    st.subheader("Suggested Questions")

    suggestions = [
        "What are the main risk drivers in my portfolio?",
        "How diversified is my portfolio?",
        "Recommend stocks to add to my portfolio for better diversification",
        "What market factors am I most exposed to?",
        "How does my portfolio perform in different market regimes?",
        "Which positions should I consider reducing for risk management?"
    ]

    for suggestion in suggestions:
        if st.button(suggestion, key=f"suggest_{suggestion[:20]}"):
            return suggestion

    return None
