import html

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_text(value: str) -> str:
    """Use the same text preparation for job titles and searches."""
    return " ".join(html.unescape(value).casefold().split())


class JobRecommender:
    def __init__(self, jobs: pd.DataFrame):
        required = {"title", "country", "published_date"}
        missing = required - set(jobs.columns)

        if missing:
            raise ValueError(f"Catalog is missing columns: {sorted(missing)}")

        catalog = jobs.copy()

        if "eligible_for_recommendation" in catalog:
            eligible = (
                catalog["eligible_for_recommendation"]
                .astype("string")
                .str.strip()
                .str.casefold()
                .eq("true")
                .fillna(False)
            )
            catalog = catalog.loc[eligible].copy()

        catalog = catalog.dropna(subset=["title"]).copy()
        catalog["title"] = catalog["title"].astype(str).map(html.unescape)
        catalog["search_title"] = catalog["title"].map(normalize_text)
        catalog = catalog.loc[catalog["search_title"].ne("")]

        self.catalog = catalog.reset_index(drop=True)

        if self.catalog.empty:
            raise ValueError("The catalog contains no eligible job titles.")

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=150_000,
            dtype=np.float32,
        )
        self.title_vectors = self.vectorizer.fit_transform(
            self.catalog["search_title"]
        )

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        columns = ["title", "country", "published_date"]
        query = normalize_text(query)

        if len(query) < 2:
            raise ValueError("Enter at least two characters.")
        if top_k < 1:
            raise ValueError("The result count must be at least one.")

        query_vector = self.vectorizer.transform([query])

        # TF-IDF vectors have unit norm, so their dot product is cosine similarity.
        scores = (self.title_vectors @ query_vector.T).toarray().ravel()
        matched = np.flatnonzero(scores > 0)

        ranked = matched[
            np.argsort(-scores[matched], kind="stable")[:top_k]
        ]

        results = self.catalog.iloc[ranked][columns].copy()
        results["similarity"] = scores[ranked].round(4)

        return results.reset_index(drop=True)
