"""Run with: streamlit run app.py."""
import streamlit as st

from loswatch.database import initialize, make_engine
from loswatch.ui import dashboard

st.set_page_config(page_title="LOSWatch-AI", page_icon="🏥", layout="wide")


@st.cache_resource
def session_factory():
    return initialize(make_engine())


dashboard(session_factory())
