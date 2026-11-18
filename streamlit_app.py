import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from src.recommender import normalize_text


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_INDEX = Path(__file__).resolve().parent / "artifacts" / "minilm_index"
BASE_COLUMNS = ["title", "country", "published_date"]
PAY_COLUMNS = ["is_hourly", "hourly_low", "hourly_high", "budget"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("project8.search")

st.set_page_config(
    page_title="Job Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)

SIDEBAR_STYLE = """
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #14244a, #204467 58%, #115650);
}
[data-testid="stSidebarNav"] a {
    background: #233f63;
    border: 1px solid #58749a;
    border-radius: 10px;
    margin-block: .2rem;
}
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] a :is(span, p),
[data-testid="stSidebarHeader"] button {
    color: #f0fbff !important;
}
[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNav"] a:focus-visible {
    background: #355881;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: #9debd8;
    border-color: #b6f7e6;
}
[data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebarNav"] a[aria-current="page"] :is(span, p) {
    color: #103b3a !important;
    font-weight: 700;
}
</style>
"""

INSIGHTS_STYLE = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(145deg, #a78bfa24, transparent 70%);
}
[data-testid="stMain"] [role="tablist"] {
    background: #a78bfa20;
    border-radius: 12px;
    padding-inline: .5rem;
}
</style>
"""

STYLE = """
<style>
[data-testid="stAppViewContainer"]:has(.st-key-job_search_page) {
    background-image: linear-gradient(155deg, #3b82f61a, #14b8a60c 55%, #8b5cf619);
}
.st-key-job_search_page [data-testid="stForm"] {
    background: #60a5fa1a;
    border: 1px solid #3b82f675;
    border-left: 4px solid #3b82f6;
    border-radius: 16px;
}
.st-key-job_search_page [data-testid="stFormSubmitButton"] button {
    background: #0e635f;
    border-color: #0e635f;
    color: #ffffff;
}
.st-key-job_search_page [data-testid="stFormSubmitButton"] button:hover {
    background: #094c49;
    border-color: #094c49;
    color: #ffffff;
}
.st-key-job_search_page .st-key-job_results {
    background: #2dd4bf1a;
    border: 1px solid #14b8a675;
    border-left: 4px solid #0d9488;
    border-radius: 16px;
    padding: 1rem;
}
.st-key-job_search_page [data-testid="stExpander"] details {
    background: #a78bfa1a;
    border: 1px solid #8b5cf675;
    border-left: 4px solid #8b5cf6;
    border-radius: 14px;
}

.st-key-job_query [data-testid="stWidgetLabel"] p {
    background: #dbeafe;
    border-radius: 6px;
    color: #1e3a8a !important;
    font-weight: 700;
    padding: .15rem .45rem;
}
.st-key-job_query [data-baseweb="input"],
.st-key-job_query input {
    background: #eff6ff;
    color: #14244a !important;
    caret-color: #14244a;
}
.st-key-job_query input::placeholder {
    color: #4b5e78 !important;
    opacity: 1;
}
.p8-hero, .p8-credit { font-family: "DM Sans", system-ui, sans-serif; }
.p8-hero {
    background: radial-gradient(ellipse at 85% 12%, #168c82, transparent 65%),
                linear-gradient(120deg, #173b83 0%, #155a8a 50%, #08665d 100%);
    color: #f6fffd;
    border: 1px solid #45aaa0;
    border-radius: 24px;
    box-shadow: 0 10px 28px #173b8326;
    padding: clamp(26px, 5vw, 54px);
    margin: 0 0 1.6rem;
}
.p8-hero .p8-kicker {
    color: #a6e8d8;
    font-weight: 700;
    letter-spacing: .16em;
    font-size: .76rem;
    margin: 0 0 .7rem;
}
.p8-hero h1 {
    color: #f6fffd;
    font-size: clamp(2rem, 5vw, 3.5rem);
    line-height: 1.12;
    padding: 0;
    margin: 0 0 .75rem;
}
.p8-hero .p8-summary {
    color: #d0f1e9;
    font-size: 1rem;
    line-height: 1.55;
    max-width: 44rem;
    margin: 0;
}
.p8-credit {
    background: #e0f2fe;
    border: 1px solid #93c5fd;
    border-radius: 10px;
    padding: .8rem 1rem;
    font-size: .9rem;
}
.p8-credit, .p8-credit strong {
    color: #14244a !important;
}
</style>
"""


@st.cache_resource(show_spinner="Loading the MiniLM search index…")
def load_search_index(dataset_repo, dataset_revision, model_revision):
    def index_file(name):
        if not dataset_repo:
            return LOCAL_INDEX / name
        return hf_hub_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            revision=dataset_revision,
            filename=name,
            token=os.environ.get("HF_TOKEN"),
        )

    catalog = pd.read_csv(index_file("catalog.csv.gz"))
    vectors = np.load(index_file("title_vectors.npy"), mmap_mode="r", allow_pickle=False)
    codes = catalog["vector_id"].to_numpy(dtype=np.intp)

    if vectors.ndim != 2 or not len(codes) or codes.min() < 0 or codes.max() >= len(vectors):
        raise ValueError("The catalog and vector files are invalid or mismatched.")

    model = SentenceTransformer(MODEL_ID, revision=model_revision)
    dimension = getattr(model, "get_embedding_dimension", None)
    if dimension is None:
        dimension = model.get_sentence_embedding_dimension
    if vectors.shape[1] != dimension():
        raise ValueError("The index does not match the model's embedding dimension.")

    logger.info("Loaded MiniLM index with %d postings", len(catalog))
    return catalog, vectors, codes, model


def advertised_pay(row):
    payment_type = str(row["is_hourly"]).strip().lower()
    low, high, budget = row["hourly_low"], row["hourly_high"], row["budget"]

    if payment_type == "true" and np.isfinite(low) and np.isfinite(high) and 0 < low <= high:
        return f"{low:,.2f}–{high:,.2f} / hr"
    if payment_type == "false" and np.isfinite(budget) and budget > 0:
        return f"{budget:,.2f} fixed"
    return "Not listed"


def upwork_url(value):
    """Allow only HTTPS links to Upwork."""
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value.strip())
        if parsed.scheme == "https" and parsed.hostname in {"upwork.com", "www.upwork.com"}:
            return value.strip()
    except ValueError:
        pass
    return None


def search_results(catalog, positions, scores):
    matches = catalog.iloc[positions].copy()
    results = matches[BASE_COLUMNS].copy()
    if set(PAY_COLUMNS).issubset(matches.columns):
        for column in PAY_COLUMNS[1:]:
            matches[column] = pd.to_numeric(matches[column], errors="coerce").astype(float)
        results["advertised_pay"] = matches.apply(advertised_pay, axis=1)
    if "link" in matches.columns:
        results["link"] = matches["link"].map(upwork_url)
    results["similarity"] = scores[positions].round(4)
    return results



def render_job_search():
    with st.container(key="job_search_page"):
        st.html(
            STYLE + """
            <section class="p8-hero">
                <p class="p8-kicker">JOB SEARCH · MARKET INSIGHTS</p>
                <h1>Job Recommender</h1>
                <p class="p8-summary">
                    Search historical freelance postings by role or skill.
                    Browse Market Insights for posting trends, advertised pay, and demand forecasts.
                </p>
            </section>
            """
        )

        st.subheader("Find relevant jobs")
        with st.form("job_search"):
            query_col, limit_col = st.columns([3, 1])
            with query_col:
                query = st.text_input("Job title or skills", placeholder="e.g. Python SQL data analyst", key="job_query")
            with limit_col:
                top_k = st.slider("Maximum results", 1, 20, 5)
            submitted = st.form_submit_button("Find jobs", type="primary")

        if submitted:
            query = normalize_text(query)
            if len(query) < 2:
                st.warning("Enter at least two characters.")
            else:
                try:
                    catalog, vectors, codes, model = load_search_index(
                        st.secrets.get("HF_DATASET_REPO", ""),
                        st.secrets.get("HF_DATASET_REVISION", ""),
                        st.secrets["MINILM_REVISION"],
                    )
                    query_vector = model.encode([query], normalize_embeddings=True)[0]
                    scores = (vectors @ query_vector)[codes]
                    positions = np.argsort(-scores, kind="stable")[:top_k]
                    results = search_results(catalog, positions, scores)

                    with st.container(key="job_results"):
                        st.subheader("Recommended postings")
                        st.caption(f"{len(results)} matches, sorted by title similarity")
                        st.dataframe(
                            results,
                            hide_index=True,
                            column_config={
                                "title": "Job title",
                                "country": "Country as recorded",
                                "published_date": "Published date",
                                "advertised_pay": "Advertised pay",
                                "similarity": st.column_config.NumberColumn("Similarity", format="%.4f"),
                                "link": st.column_config.LinkColumn(
                                    "Upwork listing", display_text="Open on Upwork"
                                ),
                            },
                        )
                        if "advertised_pay" in results.columns:
                            st.caption("Advertised pay is shown in the dataset’s original units.")
                        if "link" in results.columns:
                            st.caption("Some historical listings may have expired.")
                    logger.info("MiniLM search returned %d results", len(results))
                except Exception:
                    logger.exception("MiniLM search failed")
                    st.error("Search is temporarily unavailable. Check the app logs for details.")

        with st.expander("About this application"):
            st.write(
                "MiniLM compares the meaning of your search with historical Upwork job titles. "
                "Market Insights shows posting activity, job categories, advertised hourly rates, "
                "and an evaluation of the weekly demand forecast."
            )

        st.html(
            '<div class="p8-credit">Created by <strong>Megha Parasher</strong></div>'
        )


pages = [st.Page(render_job_search, title="Job Search", default=True)]
insights_path = Path(__file__).resolve().parent / "pages" / "Market_Insights.py"
if insights_path.is_file():
    pages.append(st.Page("pages/Market_Insights.py", title="Market Insights", url_path="Market_Insights"))

active_page = st.navigation(pages)
st.html(SIDEBAR_STYLE)
if active_page.title == "Market Insights":
    st.html(INSIGHTS_STYLE)
active_page.run()