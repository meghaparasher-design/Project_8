import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from src.recommender import JobRecommender


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("project8.recommender")

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "data" / "processed" / "jobs_features.csv"

DEMO_TITLES = [
    "Python and SQL data analyst for sales reporting",
    "Power BI dashboard developer",
    "Machine learning engineer for document classification",
    "WordPress developer for an ecommerce website",
    "React frontend developer",
    "Graphic designer for logo and brand identity",
    "Video editor for YouTube tutorials",
    "SEO specialist for ecommerce product pages",
    "Virtual assistant for customer support",
    "Bookkeeper with QuickBooks experience",
]

DISPLAY_COLUMNS = {
    "title": "Job title",
    "country": "Country as recorded",
    "published_date": "Published date",
    "similarity": "Title similarity",
}


@st.cache_resource
def load_recommender(catalog_version):
    if catalog_version is None:
        jobs = pd.DataFrame({
            "title": DEMO_TITLES,
            "country": ["—"] * len(DEMO_TITLES),
            "published_date": ["—"] * len(DEMO_TITLES),
        })
    else:
        columns = {
            "title",
            "country",
            "published_date",
            "eligible_for_recommendation",
        }
        jobs = pd.read_csv(CATALOG, usecols=lambda name: name in columns)

    engine = JobRecommender(jobs)
    logger.info("Search index ready with %d titles", len(engine.catalog))
    return engine


st.set_page_config(page_title="Project 8 Job Recommender")
st.title("Job Recommender")

catalog_version = CATALOG.stat().st_mtime_ns if CATALOG.is_file() else None

if catalog_version is None:
    st.info("Synthetic demo catalog for testing the hosted app.")
else:
    st.caption("Historical Upwork postings. Current availability may differ.")

with st.form("job_search"):
    query = st.text_input(
        "Job title or skills",
        placeholder="Python SQL data analyst",
    )
    top_k = st.slider("Maximum results", 1, 20, 5)
    submitted = st.form_submit_button("Find jobs")

if submitted:
    try:
        engine = load_recommender(catalog_version)
        results = engine.search(query, top_k=top_k)
    except ValueError as error:
        st.warning(str(error))
    else:
        logger.info("Search completed with %d results", len(results))
        if results.empty:
            st.info("No titles matched. Try a different job title or skill.")
        else:
            st.dataframe(
                results.rename(columns=DISPLAY_COLUMNS),
                hide_index=True,
            )