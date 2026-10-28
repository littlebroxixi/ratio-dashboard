"""Keep the legacy BTC page URL routed through the shared dashboard UI."""
import streamlit as st

st.session_state.page = "btc_gold"
st.switch_page("streamlit_app.py")
