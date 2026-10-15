import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Facebook Live Clustering Dashboard",
    page_icon="🌟",
    layout="wide"
)

# --- Styling ---
st.markdown("""
<style>
.stApp { background-color: #bae6fd; }
.pbi-card {
    background: #ffffff; padding: 20px 24px; border-radius: 8px;
    border-left: 6px solid #0284c7;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,.1);
    margin-bottom: 10px; transition: transform .2s ease;
}
.pbi-card:hover { transform: translateY(-2px); }
.pbi-card-title {
    font-size: 13px; text-transform: uppercase; letter-spacing: .8px;
    color: #64748b; font-weight: 700; margin-bottom: 6px;
}
.pbi-card-value { font-size: 28px; font-weight: 800; color: #0f172a; line-height: 1.2; }
.pbi-card-subtext { font-size: 12px; color: #0369a1; margin-top: 6px; font-weight: 500; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] {
    background-color: #e0f2fe; border-radius: 8px 8px 0 0;
    padding: 10px 20px; font-weight: bold; color: #0369a1;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg,#0369a1 0%,#075985 100%) !important;
    color: white !important;
}
.stButton > button, div.stFormSubmitButton > button {
    background: linear-gradient(135deg,#0284c7 0%,#1e40af 100%);
    color: white; font-weight: bold; border: none; border-radius: 8px;
    padding: .6rem 1.2rem;
}
h1, h2, h3, h4, h5, h6, p, span { color: #090d16; }
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid")

FEATURES = [
    "num_reactions", "num_comments", "num_shares",
    "num_likes", "num_loves", "num_wows",
    "num_hahas", "num_sads", "num_angrys"
]
DROP_COLS = ["status_id", "status_published", "Column1", "Column2", "Column3", "Column4"]

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        candidates = [Path(__file__).resolve().parent / "Live.csv",
                      Path.cwd() / "Live.csv"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            return None
        df = pd.read_csv(path)
    return df

st.title("🌟 Facebook Live Post Engagement Dashboard")
st.markdown(
    "Explore post-engagement patterns using **K-Means Unsupervised Clustering**, "
    "analyze engagement distributions, and predict cluster assignments for new posts."
)

with st.sidebar:
    st.header("⚙️ Model Configuration")
    uploaded_csv = st.file_uploader("Upload Live.csv dataset (optional)", type=["csv"])
    n_clusters = st.slider("Select Number of Clusters (k)", min_value=2, max_value=8, value=4)
    show_raw = st.checkbox("Preview cleaned dataset", value=False)

df = load_data(uploaded_csv)
if df is None:
    st.info("Please place `Live.csv` in the same folder as `app.py`, or upload it using the sidebar.")
    st.stop()

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error("The uploaded CSV is missing required columns: " + ", ".join(missing))
    st.stop()

data = df.copy()
data.drop(columns=[c for c in DROP_COLS if c in data.columns], inplace=True, errors="ignore")
for col in FEATURES:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data = data.dropna(subset=FEATURES).copy()

if len(data) < n_clusters:
    st.error(f"Need at least {n_clusters} valid rows for K-Means; found {len(data)}.")
    st.stop()

X = data[FEATURES].copy()
X = X.fillna(X.median())
X_log = np.log1p(X.clip(lower=0))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(X_scaled)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prediction Studio", "📈 Overview & Metrics",
    "📊 Elbow & PCA Visuals", "🔍 Raw Data Explorer", "ℹ️ About Us"
])

with tab1:
    st.subheader("🎯 Live Post Cluster Prediction Studio")
    st.write("Enter engagement statistics for a new post to estimate its cluster.")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            p_reactions = st.number_input("Num Reactions", min_value=0, value=150)
            p_comments = st.number_input("Num Comments", min_value=0, value=25)
            p_shares = st.number_input("Num Shares", min_value=0, value=5)
        with col2:
            p_likes = st.number_input("Num Likes", min_value=0, value=120)
            p_loves = st.number_input("Num Loves", min_value=0, value=20)
            p_wows = st.number_input("Num Wows", min_value=0, value=5)
        with col3:
            p_hahas = st.number_input("Num Hahas", min_value=0, value=3)
            p_sads = st.number_input("Num Sads", min_value=0, value=1)
            p_angrys = st.number_input("Num Angrys", min_value=0, value=1)
        submitted = st.form_submit_button("🔮 Predict Post Cluster")

    if submitted:
        input_df = pd.DataFrame([[
            p_reactions, p_comments, p_shares, p_likes, p_loves,
            p_wows, p_hahas, p_sads, p_angrys
        ]], columns=FEATURES)
        input_scaled = scaler.transform(np.log1p(input_df))
        predicted_cluster = int(kmeans.predict(input_scaled)[0])
        st.success(f"✨ This post is assigned to **Cluster ID: {predicted_cluster}**")
        profile = data.loc[data["Cluster"] == predicted_cluster, FEATURES].mean().sort_values(ascending=False)
        st.markdown("#### 📊 Average engagement profile for this cluster")
        st.bar_chart(profile)

with tab2:
    st.subheader("💡 Key Engagement Insights")
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Total Posts Evaluated", f"{len(data):,}", "Records in cleaned dataset"),
        ("Cluster Count (k)", str(n_clusters), "Active K-Means groups"),
        ("Average Reactions", f"{data['num_reactions'].mean():.1f}", "Per post"),
        ("Average Shares", f"{data['num_shares'].mean():.1f}", "Per post"),
    ]
    for col, (title, value, subtext) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f'<div class="pbi-card"><div class="pbi-card-title">{title}</div>'
                f'<div class="pbi-card-value">{value}</div>'
                f'<div class="pbi-card-subtext">{subtext}</div></div>',
                unsafe_allow_html=True
            )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Post counts across clusters")
        counts = data["Cluster"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#0284c7")
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Number of posts")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("#### Average reactions per cluster")
        means = data.groupby("Cluster", as_index=False)["num_reactions"].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=means, x="Cluster", y="num_reactions", ax=ax, color="#2563eb")
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Average reactions")
        st.pyplot(fig)
        plt.close(fig)

with tab3:
    st.subheader("📉 Model Diagnostics and Visualization")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Elbow Method")
        if st.button("Calculate Inertia Curve"):
            inertias = []
            max_k = min(9, len(data) - 1)
            k_range = range(1, max_k + 1)
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_scaled)
                inertias.append(km.inertia_)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(list(k_range), inertias, marker="o", color="#0369a1", linewidth=2)
            ax.set_xlabel("Number of clusters (k)")
            ax.set_ylabel("Inertia")
            ax.set_title("Elbow Curve")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Click the button to calculate and display the Elbow curve.")
    with right:
        st.markdown("#### PCA Cluster Projection")
        if len(X_scaled) >= 2:
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(pcs, columns=["PCA1", "PCA2"])
            pca_df["Cluster"] = data["Cluster"].astype(str).values
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="Cluster", palette="deep", alpha=.7, s=40, ax=ax)
            ax.set_title("PCA Dimensionality Reduction")
            st.pyplot(fig)
            plt.close(fig)

with tab4:
    st.subheader("🔍 Processed Dataset Explorer")
    selected = st.selectbox("Filter by Cluster ID", ["All"] + sorted(data["Cluster"].unique().tolist()))
    view = data if selected == "All" else data[data["Cluster"] == selected]
    st.dataframe(view if show_raw or selected != "All" else view.head(100), use_container_width=True)
    st.caption(f"Showing {len(view) if selected != 'All' else min(len(view), 100)} rows. Enable 'Preview cleaned dataset' to view the full table.")

with tab5:
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    This dashboard groups Facebook Live post records according to engagement metrics
    such as reactions, comments, shares, likes, loves, wows, hahas, sads, and angrys.

    **Algorithm:** K-Means clustering  
    **Preprocessing:** log1p transformation and StandardScaler  
    **Visualizations:** Elbow Method and PCA  
    **Dataset:** Facebook Live Sellers in Thailand (`Live.csv`)

    Cluster IDs are arbitrary group labels; they do not automatically correspond to engagement categories.
    """)

st.caption("Educational project • K-Means cluster assignments are exploratory and depend on the selected dataset and k.")
