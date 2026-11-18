import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from src.recommender import normalize_text


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_INDEX = Path(__file__).resolve().parent / "artifacts" / "minilm_index"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("project8.search")


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
    vectors = np.load(
        index_file("title_vectors.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    codes = catalog["vector_id"].to_numpy(dtype=np.intp)

    if vectors.ndim != 2 or not len(codes) or codes.min() < 0 or codes.max() >= len(vectors):
        raise ValueError("The catalog and vector files are invalid or mismatched.")

    model = SentenceTransformer(MODEL_ID, revision=model_revision)
    if vectors.shape[1] != model.get_sentence_embedding_dimension():
        raise ValueError("The index does not match the model's embedding dimension.")

    logger.info("Loaded MiniLM index with %d postings", len(catalog))
    return catalog, vectors, codes, model


st.title("Project 8 Job Recommender")
st.caption("MiniLM semantic search over historical Upwork job postings.")

with st.form("job_search"):
    query = st.text_input("Job title or skills", placeholder="Python SQL data analyst")
    top_k = st.slider("Maximum results", 1, 20, 5)
    submitted = st.form_submit_button("Find jobs")

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

            results = catalog.iloc[positions][
                ["title", "country", "published_date", "link",
                 "is_hourly", "hourly_low", "hourly_high", "budget"]
            ].copy()

            def advertised_pay(row):
                payment_type = str(row["is_hourly"]).strip().lower()
                low = row["hourly_low"]
                high = row["hourly_high"]
                budget = row["budget"]

                if payment_type == "true":
                    if pd.notna(low) and pd.notna(high) and 0 < low <= high:
                        return f"{low:,.2f}–{high:,.2f} / hr"
                elif payment_type == "false":
                    if pd.notna(budget) and budget > 0:
                        return f"{budget:,.2f} fixed"

                return "Not listed"

            results["advertised_pay"] = results.apply(advertised_pay, axis=1)
            results["similarity"] = scores[positions].round(4)

            st.dataframe(
                results[
                    ["title", "country", "published_date",
                     "advertised_pay", "similarity", "link"]
                ],
                hide_index=True,
                column_config={
                    "advertised_pay": "Advertised pay",
                    "link": st.column_config.LinkColumn(
                        "Upwork listing", display_text="Open on Upwork"
                    ),
                },
            )
            st.caption(
                "Pay is shown in the dataset’s original units. "
                "Historical Upwork listings may have expired."
            )
            logger.info("MiniLM search returned %d results", len(results))
        except Exception:
            logger.exception("MiniLM search failed")
            st.error("Search is temporarily unavailable. Check the app logs for details.")