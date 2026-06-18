import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import re

analyzer = SentimentIntensityAnalyzer()

st.set_page_config(
    page_title="DishRadar AI",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ DishRadar AI")
st.subheader("Restaurant Customer Intelligence Agent")

st.write(
    "Upload or paste customer feedback from Google Reviews, Instagram, Reddit, "
    "and competitors. Sprint 1 focuses only on collecting and organizing data."
)

restaurant_name = st.text_input("Restaurant Name")

st.divider()

col1, col2 = st.columns(2)

with col1:
    google_reviews = st.text_area("Google Reviews", height=200)
    instagram_comments = st.text_area("Instagram Comments", height=200)

with col2:
    reddit_mentions = st.text_area("Reddit Mentions", height=200)
    competitor_reviews = st.text_area("Competitor Reviews", height=200)


def text_to_rows(text, source, restaurant_name, is_competitor=False):
    rows = []

    if not text.strip():
        return rows

    lines = text.split("\n")

    for line in lines:
        clean_line = line.strip()

        if clean_line:
            rows.append({
                "restaurant_name": restaurant_name,
                "source": source,
                "text": clean_line,
                "is_competitor": is_competitor
            })

    return rows


def get_sentiment(text):

    score = analyzer.polarity_scores(
        str(text)
    )["compound"]

    if score >= 0.05:
        label = "Positive"

    elif score <= -0.05:
        label = "Negative"

    else:
        label = "Neutral"

    return score, label


THEME_KEYWORDS = {
    "Food Quality": ["food", "taste", "flavor", "fresh", "delicious", "cold", "stale", "bland"],
    "Service": ["service", "staff", "waiter", "server", "rude", "friendly", "helpful"],
    "Wait Time": ["wait", "waiting", "slow", "delayed", "forever", "late"],
    "Price": ["price", "expensive", "cheap", "overpriced", "value"],
    "Portion Size": ["portion", "small", "large", "quantity", "serving"],
    "Atmosphere": ["atmosphere", "ambience", "vibe", "music", "decor"],
    "Cleanliness": ["clean", "dirty", "bathroom", "hygiene"],
    "Parking": ["parking", "valet", "park"],
}

THEME_PATTERNS = {
    theme: re.compile(
        r"\b(" + "|".join(re.escape(kw) for kw in keywords) + r")\b",
        flags=re.IGNORECASE
    )
    for theme, keywords in THEME_KEYWORDS.items()
}


def extract_themes(df: pd.DataFrame, sentiment_filter: str) -> pd.DataFrame:
    if "sentiment_label" not in df.columns or "text" not in df.columns:
        return pd.DataFrame(columns=["Theme", "Mentions"])

    if sentiment_filter not in df["sentiment_label"].unique():
        return pd.DataFrame(columns=["Theme", "Mentions"])

    filtered_texts = (
        df.loc[df["sentiment_label"] == sentiment_filter, "text"]
        .astype(str)
        .tolist()
    )

    theme_counts = {
        theme: sum(1 for text in filtered_texts if pattern.search(text))
        for theme, pattern in THEME_PATTERNS.items()
    }

    theme_counts = {theme: count for theme,
                    count in theme_counts.items() if count > 0}

    return (
        pd.DataFrame(theme_counts.items(), columns=["Theme", "Mentions"])
        .sort_values("Mentions", ascending=False)
        .reset_index(drop=True)
    )


if st.button("Analyze Feedback"):
    all_rows = []

    all_rows.extend(text_to_rows(
        google_reviews, "Google Reviews", restaurant_name))
    all_rows.extend(text_to_rows(instagram_comments,
                    "Instagram", restaurant_name))
    all_rows.extend(text_to_rows(reddit_mentions, "Reddit", restaurant_name))
    all_rows.extend(text_to_rows(competitor_reviews,
                    "Competitor Reviews", restaurant_name, True))

    df = pd.DataFrame(all_rows)

    if not df.empty:
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"] != ""]

    if not df.empty:
        df[["sentiment_score", "sentiment_label"]] = df["text"].apply(
            lambda x: pd.Series(get_sentiment(x))
        )

    if df.empty:
        st.warning("Please paste at least one review or comment.")
    else:
        st.success("Feedback data loaded successfully.")
        st.metric("Total Feedback Items", len(df))
        sentiment_counts = df["sentiment_label"].value_counts()

        col1, col2, col3 = st.columns(3)

        col1.metric("Positive", sentiment_counts.get("Positive", 0))
        col2.metric("Neutral", sentiment_counts.get("Neutral", 0))
        col3.metric("Negative", sentiment_counts.get("Negative", 0))

        sentiment_df = sentiment_counts.reset_index()
        sentiment_df.columns = ["Sentiment", "Count"]

        fig = px.pie(
            sentiment_df,
            names="Sentiment",
            values="Count",
            title="Sentiment Breakdown"
        )

        st.plotly_chart(fig, use_container_width=True)

        praise_df = extract_themes(df, "Positive")
        complaint_df = extract_themes(df, "Negative")

        st.subheader("Top Praise Themes")
        st.dataframe(praise_df, use_container_width=True)

        st.subheader("Top Complaint Themes")
        st.dataframe(complaint_df, use_container_width=True)

        st.subheader("Sentiment Summary")
        st.dataframe(sentiment_df, use_container_width=True)

        st.subheader("Processed Feedback")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Processed Data as CSV",
            data=csv,
            file_name="dishradar_feedback_data.csv",
            mime="text/csv"
        )
