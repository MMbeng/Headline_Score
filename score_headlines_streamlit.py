"""Streamlit client for headline sentiment scoring."""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

DEFAULT_BACKEND_URL = "http://localhost:8013"


def init_session() -> None:
    if "rows" not in st.session_state:
        st.session_state.rows = [
            "Stocks sink on inflation jitters",
            "Company beats earnings expectations",
            "Oil prices fall amid weak demand",
        ]
    if "results" not in st.session_state:
        st.session_state.results = None
    if "used_rows" not in st.session_state:
        st.session_state.used_rows = None


def sidebar_controls() -> str:
    st.sidebar.header("Backend")
    backend_url = st.sidebar.text_input("FastAPI URL", value=DEFAULT_BACKEND_URL)
    col_left, col_right = st.sidebar.columns(2)
    if col_left.button("Check API"):
        try:
            response = requests.get(f"{backend_url}/status", timeout=5)
            if response.ok and response.json().get("status") == "OK":
                st.sidebar.success("API OK")
            else:
                st.sidebar.error("API reachable, bad response")
        except requests.exceptions.RequestException as exc:
            st.sidebar.error(f"Failed: {exc}")
    if col_right.button("Reset App"):
        st.session_state.rows = []
        st.session_state.results = None
        st.session_state.used_rows = None
        st.toast("Reset")
    return backend_url


def bulk_paste_controls() -> None:
    with st.expander("Quick paste"):
        pasted = st.text_area("One headline per line")
        col_left, col_right = st.columns(2)
        if col_left.button("Replace from paste"):
            lines = [x.strip() for x in pasted.splitlines() if x.strip()]
            st.session_state.rows = lines
            st.session_state.results = None
            st.session_state.used_rows = None
        if col_right.button("Load sample"):
            st.session_state.rows = [
                "Fed hints at holding rates steady",
                "Tech shares rally after earnings beat",
                "Geopolitical tensions rattle markets",
                "Retail sales surprise to the upside",
            ]
            st.session_state.results = None
            st.session_state.used_rows = None


def editable_rows() -> None:
    st.subheader("Headlines")
    updated_rows: List[str] = []
    for index, text_value in enumerate(st.session_state.rows):
        col_text, col_del = st.columns([10, 1])
        new_value = col_text.text_input(f"Headline {index + 1}", value=text_value, key=f"row_{index}")
        if not col_del.button("✕", key=f"del_{index}"):
            updated_rows.append(new_value)
    st.session_state.rows = updated_rows


def add_clear_buttons() -> Tuple[bool, bool]:
    col_add, col_clear, _ = st.columns([1, 1, 6])
    add_clicked = col_add.button("Add headline")
    clear_clicked = col_clear.button("Clear all")
    if add_clicked:
        st.session_state.rows.append("")
    if clear_clicked:
        st.session_state.rows = []
        st.session_state.results = None
        st.session_state.used_rows = None
    return add_clicked, clear_clicked


def score(backend_url: str, rows: List[str]) -> Optional[List[str]]:
    payload_rows = [r.strip() for r in rows if r.strip()]
    if not payload_rows:
        st.warning("Add at least one non-empty headline.")
        return None
    try:
        with st.spinner("Scoring…"):
            response = requests.post(
                f"{backend_url}/score_headlines",
                data=json.dumps({"headlines": payload_rows}),
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
        if not response.ok:
            st.error(f"API error {response.status_code}: {response.text}")
            return None
        labels = response.json().get("labels", [])
        st.session_state.used_rows = payload_rows
        st.session_state.results = labels
        st.toast("Scored")
        return labels
    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")
        return None


def results_table() -> None:
    if not (st.session_state.results and st.session_state.used_rows):
        return
    df_result = pd.DataFrame({"headline": st.session_state.used_rows, "label": st.session_state.results})
    st.subheader("Results")
    st.dataframe(df_result, use_container_width=True, height=320)
    col_total, col_opt, col_pess = st.columns(3)
    value_counts = df_result["label"].value_counts()
    col_total.metric("Total", int(value_counts.sum()))
    col_opt.metric("Optimistic", int((df_result["label"] == "Optimistic").sum()))
    col_pess.metric("Pessimistic", int((df_result["label"] == "Pessimistic").sum()))
    csv_bytes = df_result.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="headline_sentiment.csv", mime="text/csv")


def main() -> None:
    st.set_page_config(page_title="Headline Sentiment Lab", page_icon="🗞️", layout="wide")
    init_session()
    backend_url = sidebar_controls()
    st.title("🗞️ Headline Sentiment Lab")
    st.caption("Add, edit, or remove headlines. Score them via your FastAPI service on port 8013.")
    bulk_paste_controls()
    editable_rows()
    _, _, score_col = st.columns([1, 1, 3])
    add_clear_buttons()
    if score_col.button("Score headlines", type="primary"):
        score(backend_url, st.session_state.rows)
    results_table()


if __name__ == "__main__":
    main()
