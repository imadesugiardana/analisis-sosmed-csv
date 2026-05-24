# app.py — Dashboard Analitik Media Sosial dengan Streamlit


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
from datetime import datetime

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Social Media Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>
        .main {
            background-color: #0f172a;
        }

        .stMetric {
            background-color: #1e293b;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #334155;
        }

        h1, h2, h3 {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# HEADER
# ==========================================

st.title("📊 Social Media Analytics Dashboard")
st.markdown(
    "Dashboard analitik media sosial berbasis AI menggunakan Streamlit"
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("⚙️ Pengaturan")

uploaded_file = st.sidebar.file_uploader(
    "Upload File CSV",
    type=["csv"]
)

# ==========================================
# LOAD DATA
# ==========================================

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data(file):

    # baca semua sebagai string dulu
    df = pd.read_csv(
        file,
        dtype=str
    )

    return df


if uploaded_file is None:
    st.info("Silakan upload file CSV terlebih dahulu")
    st.stop()

try:
    df = load_data(uploaded_file)

except Exception as e:
    st.error(f"Gagal membaca file CSV: {e}")
    st.stop()

# ==========================================
# DATA PREPROCESSING
# ==========================================

numeric_columns = [
    'diggCount',
    'shareCount',
    'playCount',
    'commentCount',
    'collectCount',
    'videoMeta.duration'
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        ).fillna(0)

# ==========================================
# ENGAGEMENT CALCULATION
# ==========================================

if all(col in df.columns for col in [
    'diggCount',
    'commentCount',
    'shareCount',
    'collectCount'
]):

    df['engagement_total'] = (
        df['diggCount'] +
        df['commentCount'] +
        df['shareCount'] +
        df['collectCount']
    )

if 'playCount' in df.columns:
    df['engagement_rate'] = np.where(
        df['playCount'] > 0,
        (df['engagement_total'] / df['playCount']) * 100,
        0
    )

# ==========================================
# SENTIMENT ANALYSIS
# ==========================================

@st.cache_data

def get_sentiment(text):
    try:
        polarity = TextBlob(str(text)).sentiment.polarity

        if polarity > 0:
            return 'Positive'
        elif polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'
    except:
        return 'Neutral'

if 'text' in df.columns:
    df['sentiment'] = df['text'].astype(str).apply(get_sentiment)

# ==========================================
# KPI SECTION
# ==========================================

st.header("📈 Executive Summary")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Video",
        f"{len(df):,}"
    )

with col2:
    total_views = int(df['playCount'].sum())

    st.metric(
        "Total Views",
        f"{total_views:,}"
    )

with col3:
    total_likes = int(df['diggCount'].sum())

    st.metric(
        "Total Likes",
        f"{total_likes:,}"
    )

with col4:
    total_comments = int(df['commentCount'].sum())

    st.metric(
        "Total Comments",
        f"{total_comments:,}"
    )

with col5:
    avg_engagement = round(df['engagement_rate'].mean(), 2)

    st.metric(
        "Avg Engagement",
        f"{avg_engagement}%"
    )

# ==========================================
# DESCRIPTIVE STATISTICS
# ==========================================

st.header("📋 Statistik Deskriptif")

stats_df = df[[
    'playCount',
    'diggCount',
    'commentCount',
    'shareCount',
    'collectCount',
    'engagement_rate'
]].describe()

st.dataframe(stats_df, use_container_width=True)

# ==========================================
# TOP VIRAL CONTENT
# ==========================================

st.header("🔥 Top Viral Content")

if 'authorMeta.name' in df.columns:

    top_videos = df.nlargest(10, 'playCount')

    fig_top = px.bar(
        top_videos,
        x='playCount',
        y='authorMeta.name',
        orientation='h',
        color='engagement_rate',
        title='Top 10 Video Berdasarkan Views',
        hover_data=['diggCount', 'commentCount']
    )

    fig_top.update_layout(height=500)

    st.plotly_chart(fig_top, use_container_width=True)

# ==========================================
# ENGAGEMENT SCATTER
# ==========================================

st.header("🎯 Analisis Engagement")

fig_scatter = px.scatter(
    df,
    x='playCount',
    y='engagement_rate',
    size='diggCount',
    color='commentCount',
    hover_name='authorMeta.name',
    title='Engagement vs Views'
)

st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# TREND POSTING
# ==========================================

if 'posting_date' in df.columns:

    st.header("📅 Trend Posting")

    trend_df = df.groupby('posting_date').agg({
        'playCount': 'sum',
        'diggCount': 'sum',
        'commentCount': 'sum'
    }).reset_index()

    fig_trend = px.line(
        trend_df,
        x='posting_date',
        y=['playCount', 'diggCount', 'commentCount'],
        title='Trend Aktivitas Media Sosial'
    )

    st.plotly_chart(fig_trend, use_container_width=True)

# ==========================================
# POSTING HOUR ANALYSIS
# ==========================================

if 'posting_hour' in df.columns:

    st.header("⏰ Analisis Jam Posting")

    hour_df = df.groupby('posting_hour').agg({
        'playCount': 'mean',
        'engagement_rate': 'mean'
    }).reset_index()

    fig_hour = px.bar(
        hour_df,
        x='posting_hour',
        y='engagement_rate',
        title='Engagement Berdasarkan Jam Posting'
    )

    st.plotly_chart(fig_hour, use_container_width=True)

# ==========================================
# WORD CLOUD
# ==========================================
st.header("☁️ Word Cloud Caption")

if 'text' in df.columns:

    # Bersihkan data null/kosong
    text_series = (
        df['text']
        .fillna('')
        .astype(str)
    )

    # Ambil hanya text valid
    text_list = [
        str(x)
        for x in text_series
        if str(x).strip() != ''
    ]

    all_text = ' '.join(text_list)

    # Cleaning text
    clean_text = re.sub(r'http\\S+', '', all_text)
    clean_text = re.sub(r'[^A-Za-z0-9# ]+', '', clean_text)

    if clean_text.strip():

        wordcloud = WordCloud(
            width=1200,
            height=500,
            background_color='white'
        ).generate(clean_text)

        fig_wc, ax = plt.subplots(figsize=(15, 5))

        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')

        st.pyplot(fig_wc)

    else:
        st.warning("Tidak ada teks valid untuk Word Cloud")

# ==========================================
# HASHTAG ANALYSIS
# ==========================================

st.header("#️⃣ Analisis Hashtag")

if 'text' in df.columns:

    hashtags = []

    # Bersihkan data text
    text_series = (
        df['text']
        .fillna('')
        .astype(str)
    )

    for text in text_series:

        # Pastikan benar-benar string
        safe_text = str(text)

        # Cari hashtag
        found = re.findall(r'#(\\w+)', safe_text)

        hashtags.extend(found)

    hashtag_df = pd.DataFrame(
        hashtags,
        columns=['hashtag']
    )

    if not hashtag_df.empty:

        hashtag_count = (
            hashtag_df['hashtag']
            .value_counts()
            .head(15)
            .reset_index()
        )

        hashtag_count.columns = ['hashtag', 'count']

        fig_hash = px.bar(
            hashtag_count,
            x='hashtag',
            y='count',
            title='Top Hashtag'
        )

        st.plotly_chart(
            fig_hash,
            use_container_width=True
        )

    else:
        st.warning("Tidak ditemukan hashtag")

# ==========================================
# SENTIMENT ANALYSIS VISUALIZATION
# ==========================================

st.header("😊 Sentiment Analysis")

if 'sentiment' in df.columns:

    sentiment_count = (
        df['sentiment']
        .value_counts()
        .reset_index()
    )

    sentiment_count.columns = ['sentiment', 'count']

    fig_sentiment = px.pie(
        sentiment_count,
        names='sentiment',
        values='count',
        title='Distribusi Sentimen'
    )

    st.plotly_chart(fig_sentiment, use_container_width=True)

# ==========================================
# MUSIC ANALYSIS
# ==========================================

if 'musicMeta.musicName' in df.columns:

    st.header("🎵 Analisis Music")

    music_df = (
        df.groupby('musicMeta.musicName')
        .agg({
            'playCount': 'sum',
            'engagement_rate': 'mean'
        })
        .reset_index()
        .sort_values(by='playCount', ascending=False)
        .head(10)
    )

    fig_music = px.bar(
        music_df,
        x='musicMeta.musicName',
        y='playCount',
        color='engagement_rate',
        title='Top Music Berdasarkan Views'
    )

    st.plotly_chart(fig_music, use_container_width=True)

# ==========================================
# CREATOR ANALYSIS
# ==========================================

if 'authorMeta.name' in df.columns:

    st.header("👤 Analisis Creator")

    creator_df = (
        df.groupby('authorMeta.name')
        .agg({
            'playCount': 'sum',
            'engagement_rate': 'mean',
            'diggCount': 'sum'
        })
        .reset_index()
        .sort_values(by='playCount', ascending=False)
        .head(10)
    )

    fig_creator = px.scatter(
        creator_df,
        x='playCount',
        y='engagement_rate',
        size='diggCount',
        hover_name='authorMeta.name',
        title='Performance Creator'
    )

    st.plotly_chart(fig_creator, use_container_width=True)

# ==========================================
# AI INSIGHT SECTION
# ==========================================

st.header("🤖 AI Insight")

insights = []

# Insight engagement
high_engagement = df['engagement_rate'].mean()

if high_engagement > 10:
    insights.append(
        "Konten memiliki engagement sangat tinggi secara keseluruhan."
    )
else:
    insights.append(
        "Engagement rata-rata masih dapat ditingkatkan dengan optimasi konten."
    )

# Insight durasi video
if 'videoMeta.duration' in df.columns:

    short_video = df[
        df['videoMeta.duration'] <= 20
    ]['engagement_rate'].mean()

    long_video = df[
        df['videoMeta.duration'] > 20
    ]['engagement_rate'].mean()

    if short_video > long_video:
        insights.append(
            "Video berdurasi pendek memiliki engagement lebih tinggi."
        )
    else:
        insights.append(
            "Video berdurasi panjang memiliki engagement lebih tinggi."
        )

# Insight posting hour
if 'posting_hour' in df.columns:

    best_hour = (
        df.groupby('posting_hour')['engagement_rate']
        .mean()
        .idxmax()
    )

    insights.append(
        f"Jam posting terbaik adalah sekitar pukul {best_hour}:00."
    )

# Insight sentiment
if 'sentiment' in df.columns:

    top_sentiment = df['sentiment'].value_counts().idxmax()

    insights.append(
        f"Mayoritas percakapan memiliki sentimen {top_sentiment.lower()}."
    )

for idx, insight in enumerate(insights, start=1):
    st.success(f"{idx}. {insight}")

# ==========================================
# DETAIL TABLE
# ==========================================

st.header("🗂️ Detail Data")

show_columns = [
    'authorMeta.name',
    'text',
    'playCount',
    'diggCount',
    'commentCount',
    'shareCount',
    'engagement_rate'
]

available_columns = [
    col for col in show_columns
    if col in df.columns
]

st.dataframe(
    df[available_columns],
    use_container_width=True,
    height=500
)

# ==========================================
# DOWNLOAD CSV
# ==========================================

st.header("⬇️ Download Data")

csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label='Download Processed CSV',
    data=csv,
    file_name='processed_social_media_data.csv',
    mime='text/csv'
)

# ==========================================
# FOOTER
# ==========================================

st.markdown('---')
st.caption('Developed with ❤️ using Streamlit + Python + AI Analytics')

