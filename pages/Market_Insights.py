from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Market Insights", layout="wide")
st.title("Market Insights")
st.caption("Historical Upwork postings from the Project 8 dataset.")


def load_report(relative_path):
    path = REPORTS / relative_path
    if not path.is_file():
        st.error(f"Missing report file: reports/{relative_path}")
        st.stop()
    return pd.read_csv(path)


daily_tab, category_tab, pay_tab, forecast_tab = st.tabs(
    ["Posting activity", "Job categories", "Hourly rates", "Demand forecast"]
)

with daily_tab:
    daily = load_report("step3/daily_postings.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    st.subheader("Recorded postings by day")
    st.line_chart(daily.sort_values("date").set_index("date")["postings"])
    st.caption(
        "Collection gaps and partial days can affect the recorded totals."
    )

with category_tab:
    categories = load_report("step4/category_counts.csv")
    st.subheader("Postings by title category")
    st.bar_chart(categories.set_index("job_category")["postings"])
    st.dataframe(categories, hide_index=True)
    st.caption(
        "Categories come from title rules. Multiple matches and "
        "uncategorized postings are shown separately."
    )

with pay_tab:
    country_pay = load_report("step3/country_hourly_summary.csv")
    st.subheader("Advertised hourly rates by country")
    min_postings = st.number_input(
        "Minimum postings with usable rates", min_value=1, value=100, step=1
    )
    metric = st.selectbox(
        "Show",
        ["Median hourly midpoint", "Mean hourly midpoint"],
    )
    metric_column = {
        "Median hourly midpoint": "median_hourly_midpoint",
        "Mean hourly midpoint": "mean_hourly_midpoint",
    }[metric]

    shown = country_pay.loc[
        country_pay["postings_with_pay"].ge(min_postings)
    ].nlargest(15, "postings_with_pay")

    if shown.empty:
        st.info("Lower the minimum to see countries with fewer postings.")
    else:
        st.bar_chart(shown.set_index("country_as_recorded")[metric_column])
        st.dataframe(shown.round(2), hide_index=True)
    st.caption(
        "Shows up to 15 countries by sample size. An hourly midpoint is the "
        "middle of the advertised pay range. Currency and the meaning of "
        "the country field are unverified."
    )

with forecast_tab:
    forecasts = load_report("step5/test_predictions.csv")
    forecasts["date"] = pd.to_datetime(forecasts["date"], utc=True)
    category = st.selectbox(
        "Job category", sorted(forecasts["category"].dropna().unique())
    )
    selected = forecasts.loc[forecasts["category"].eq(category)].sort_values("date")
    mae = (selected["actual"] - selected["forecast"]).abs().mean()

    st.subheader("Actual versus forecast posting counts")
    st.metric("Test MAE for this category", f"{mae:.2f} postings per day")
    st.line_chart(selected.set_index("date")[["actual", "forecast"]])
    st.caption(
        "Held-out test period: 17–23 March 2024 (UTC). Each forecast "
        "uses the recorded count from the same weekday one week earlier."
    )