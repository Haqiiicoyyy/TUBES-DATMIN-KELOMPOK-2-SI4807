"""
🥛 Milk Quality Dashboard — Streamlit App
Tugas Besar BBK2LAB3 Penambangan Data — Telkom University
Metode: K-Means Clustering + Logistic Regression + Naïve Bayes
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, silhouette_score
)
from sklearn.decomposition import PCA

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🥛 Milk Quality Dashboard",
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #e63946;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .grade-high   { color: #2ecc71; font-weight: bold; }
    .grade-medium { color: #f39c12; font-weight: bold; }
    .grade-low    { color: #e74c3c; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f0f0;
        border-radius: 8px 8px 0 0;
        padding: 6px 18px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🥛 Milk Quality Prediction Dashboard</h1>
    <p style="opacity:0.8;">BBK2LAB3 Penambangan Data — Telkom University | Semester Genap TA 2025/2026</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING  (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("milknew.csv")
    df.columns = df.columns.str.strip()          # hapus spasi nama kolom
    df.dropna(inplace=True)
    # Duplikat DIPERTAHANKAN — setiap baris adalah catatan validasi kualitas susu yang sah

    le = LabelEncoder()
    df["Grade_enc"] = le.fit_transform(df["Grade"])  # high=0, low=1, medium=2

    features = ["pH", "Temprature", "Taste", "Odor", "Fat", "Turbidity", "Colour"]
    X = df[features]
    y = df["Grade_enc"]

    scaler = MinMaxScaler()
    X_mm = pd.DataFrame(scaler.fit_transform(X), columns=features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_mm, y, test_size=0.2, random_state=42, stratify=y
    )
    return df, X, y, X_mm, X_train, X_test, y_train, y_test, scaler, le, features


@st.cache_resource
def train_models(X_mm, y, X_train, X_test, y_train, y_test):
    # ── K-Means ──────────────────────────────────────────────
    km = KMeans(n_clusters=3, n_init=20, init="k-means++", random_state=42)
    cluster_labels = km.fit_predict(X_mm)
    sil = silhouette_score(X_mm, cluster_labels)

    # ── Logistic Regression ───────────────────────────────────
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                             multi_class="auto", random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    cv_lr  = cross_val_score(lr, X_mm, y, cv=5, scoring="accuracy")

    # ── Naïve Bayes ────────────────────────────────────────────
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    cv_nb  = cross_val_score(nb, X_mm, y, cv=5, scoring="accuracy")

    return (km, cluster_labels, sil,
            lr, y_pred_lr, acc_lr, cv_lr,
            nb, y_pred_nb, acc_nb, cv_nb)


# ── Load data & train ──────────────────────────────────────────────────────────
df, X, y, X_mm, X_train, X_test, y_train, y_test, scaler, le, features = load_and_preprocess()
(km, cluster_labels, sil,
 lr, y_pred_lr, acc_lr, cv_lr,
 nb, y_pred_nb, acc_nb, cv_nb) = train_models(X_mm, y, X_train, X_test, y_train, y_test)

GRADE_COLOR = {"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Telkom_University_logo.svg/320px-Telkom_University_logo.svg.png",
             width=180)
    st.markdown("### ⚙️ Navigasi")
    page = st.radio(
        "Pilih Halaman",
        ["📊 Dataset Overview",
         "🔍 Eksplorasi Data",
         "🔵 K-Means Clustering",
         "📈 Logistic Regression",
         "🟡 Naïve Bayes",
         "🏆 Perbandingan Model",
         "🔮 Prediksi Baru"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("""
    **Dataset:** Milk Quality  
    **Sumber:** Kaggle  
    **Baris:** {:,} | **Fitur:** {}  
    **Target:** Grade (low/medium/high)
    """.format(len(df), len(features)))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.header("📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data", f"{len(df):,}")
    c2.metric("Fitur Input", len(features))
    c3.metric("Kelas Target", df["Grade"].nunique())
    c4.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("🗂️ Sample Data (10 baris pertama)")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("📐 Statistik Deskriptif")
    st.dataframe(df[features].describe().round(3), use_container_width=True)

    st.subheader("🎯 Distribusi Kelas (Grade)")
    col1, col2 = st.columns([1, 1])
    grade_counts = df["Grade"].value_counts()

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = [GRADE_COLOR[g] for g in grade_counts.index]
        bars = ax.bar(grade_counts.index, grade_counts.values, color=colors,
                      edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, grade_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    str(val), ha="center", fontweight="bold")
        ax.set_title("Distribusi Grade", fontweight="bold")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(grade_counts.values, labels=grade_counts.index,
               colors=colors, autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_title("Proporsi Grade", fontweight="bold")
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EKSPLORASI DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Eksplorasi Data":
    st.header("🔍 Eksplorasi Data (EDA)")

    tab1, tab2, tab3 = st.tabs(["📦 Distribusi Fitur", "🌡️ Korelasi", "📉 Boxplot per Grade"])

    with tab1:
        st.subheader("Histogram Fitur Numerik")
        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        axes = axes.flatten()
        for i, col in enumerate(features):
            axes[i].hist(df[col], bins=20, color="#3498db", edgecolor="white", alpha=0.85)
            axes[i].set_title(col, fontweight="bold")
        axes[-1].set_visible(False)
        plt.suptitle("Distribusi Fitur", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        st.subheader("Heatmap Korelasi")
        fig, ax = plt.subplots(figsize=(9, 7))
        corr = df[features + ["Grade_enc"]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", linewidths=0.5, ax=ax, vmin=-1, vmax=1)
        ax.set_title("Korelasi antar Fitur & Grade", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab3:
        st.subheader("Boxplot Fitur per Grade")
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        num_cols = ["pH", "Temprature", "Colour"]
        cat_cols = ["Taste", "Odor", "Fat", "Turbidity"]
        plot_cols = num_cols + cat_cols[:3]
        for i, col in enumerate(plot_cols):
            for grade in ["high", "medium", "low"]:
                data = df[df["Grade"] == grade][col]
                axes[i].boxplot(data, positions=[["high","medium","low"].index(grade)],
                                patch_artist=True,
                                boxprops=dict(facecolor=GRADE_COLOR[grade], alpha=0.7),
                                medianprops=dict(color="black", linewidth=2),
                                whiskerprops=dict(color=GRADE_COLOR[grade]),
                                capprops=dict(color=GRADE_COLOR[grade]),
                                flierprops=dict(marker="o", markerfacecolor=GRADE_COLOR[grade], markersize=4))
            axes[i].set_xticks([0, 1, 2])
            axes[i].set_xticklabels(["High", "Medium", "Low"])
            axes[i].set_title(f"Distribusi {col} per Grade", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵 K-Means Clustering":
    st.header("🔵 K-Means Clustering")

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Cluster (k)", "3")
    c2.metric("Silhouette Score", f"{sil:.4f}")
    c3.metric("Algoritma Inisialisasi", "k-means++")

    tab1, tab2, tab3 = st.tabs(["📉 Elbow Method", "🗺️ Visualisasi PCA", "📋 Profil Cluster"])

    with tab1:
        st.subheader("Elbow Method — Menentukan k Optimal")
        inertias = []
        k_range = range(2, 11)
        for k in k_range:
            km_tmp = KMeans(n_clusters=k, n_init=10, random_state=42)
            km_tmp.fit(X_mm)
            inertias.append(km_tmp.inertia_)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(list(k_range), inertias, "o-", color="#3498db", linewidth=2, markersize=8)
        ax.axvline(3, color="#e74c3c", linestyle="--", label="k=3 (dipilih)")
        ax.set_xlabel("Jumlah Cluster (k)"); ax.set_ylabel("Inertia (WCSS)")
        ax.set_title("Elbow Method", fontweight="bold")
        ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        st.subheader("Visualisasi Cluster — PCA 2D")
        pca2 = PCA(n_components=2, random_state=42)
        X_pca = pca2.fit_transform(X_mm)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        cluster_colors = ["#e74c3c", "#3498db", "#2ecc71"]
        grade_colors_map = {"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"}

        # Plot cluster
        for c in range(3):
            mask = cluster_labels == c
            axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                            c=cluster_colors[c], label=f"Cluster {c}",
                            edgecolors="white", linewidth=0.5, s=60, alpha=0.85)
        centers_pca = pca2.transform(km.cluster_centers_)
        axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1],
                        c="black", marker="X", s=200, label="Centroid", zorder=5)
        axes[0].set_title("K-Means Clustering (PCA 2D)", fontweight="bold")
        axes[0].set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
        axes[0].set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
        axes[0].legend()

        # Plot grade asli
        for grade, color in grade_colors_map.items():
            mask = df["Grade"].values == grade
            axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                            c=color, label=grade.capitalize(),
                            edgecolors="white", linewidth=0.5, s=60, alpha=0.85)
        axes[1].set_title("Grade Asli (PCA 2D)", fontweight="bold")
        axes[1].set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
        axes[1].set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
        axes[1].legend()

        plt.suptitle("Perbandingan Cluster vs Grade Asli", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab3:
        st.subheader("Profil Rata-rata Fitur per Cluster")
        df_cluster = X_mm.copy()
        df_cluster["Cluster"] = cluster_labels
        df_cluster["Grade"]   = df["Grade"].values
        cluster_profile = df_cluster.groupby("Cluster")[features].mean().round(3)
        st.dataframe(cluster_profile, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(cluster_profile, annot=True, fmt=".3f", cmap="coolwarm",
                    linewidths=0.5, ax=ax, vmin=0, vmax=1)
        ax.set_title("Heatmap Profil Cluster (MinMax Scaled)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.subheader("Cross-tabulation Cluster vs Grade Asli")
        cross_tab = pd.crosstab(cluster_labels, df["Grade"],
                                rownames=["Cluster"], colnames=["Grade Asli"])
        st.dataframe(cross_tab, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Logistic Regression":
    st.header("📈 Logistic Regression")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", f"{acc_lr:.4f}")
    c2.metric("CV Accuracy (5-fold)", f"{cv_lr.mean():.4f}")
    c3.metric("CV Std Dev", f"± {cv_lr.std():.4f}")

    tab1, tab2, tab3 = st.tabs(["📊 Confusion Matrix", "🔄 Cross Validation", "📐 Koefisien Fitur"])

    with tab1:
        st.subheader("Confusion Matrix & Classification Report")
        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_test, y_pred_lr)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=le.classes_)
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title("Confusion Matrix — Logistic Regression", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col2:
            report = classification_report(y_test, y_pred_lr,
                                           target_names=le.classes_, output_dict=True)
            report_df = pd.DataFrame(report).T.round(3)
            st.dataframe(report_df, use_container_width=True)

    with tab2:
        st.subheader("5-Fold Cross Validation Accuracy")
        fig, ax = plt.subplots(figsize=(7, 4))
        fold_labels = [f"Fold {i+1}" for i in range(len(cv_lr))]
        bars = ax.bar(fold_labels, cv_lr, color="#3498db", edgecolor="white")
        ax.axhline(cv_lr.mean(), color="#e74c3c", linestyle="--",
                   label=f"Mean = {cv_lr.mean():.4f}")
        ax.set_ylim(0, 1.15)
        ax.set_title("5-Fold CV Accuracy — Logistic Regression", fontweight="bold")
        ax.set_ylabel("Accuracy"); ax.legend()
        for bar, val in zip(bars, cv_lr):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab3:
        st.subheader("Koefisien Fitur per Kelas")
        coef_df = pd.DataFrame(
            lr.coef_, columns=features,
            index=[f"Class {le.inverse_transform([i])[0]}" for i in range(3)]
        ).T.round(4)
        st.dataframe(coef_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        coef_df.plot(kind="bar", ax=ax, edgecolor="white", width=0.65)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Koefisien Fitur — Logistic Regression", fontweight="bold")
        ax.set_xlabel("Fitur"); ax.set_ylabel("Koefisien")
        ax.set_xticklabels(features, rotation=30, ha="right")
        ax.legend(title="Kelas")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — NAÏVE BAYES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🟡 Naïve Bayes":
    st.header("🟡 Naïve Bayes Classifier")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", f"{acc_nb:.4f}")
    c2.metric("CV Accuracy (5-fold)", f"{cv_nb.mean():.4f}")
    c3.metric("CV Std Dev", f"± {cv_nb.std():.4f}")

    tab1, tab2 = st.tabs(["📊 Confusion Matrix", "🔄 Cross Validation"])

    with tab1:
        st.subheader("Confusion Matrix & Classification Report")
        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            cm_nb = confusion_matrix(y_test, y_pred_nb)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_nb,
                                          display_labels=le.classes_)
            disp.plot(ax=ax, cmap="Oranges", colorbar=False)
            ax.set_title("Confusion Matrix — Naïve Bayes", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col2:
            report_nb = classification_report(y_test, y_pred_nb,
                                               target_names=le.classes_,
                                               output_dict=True)
            report_nb_df = pd.DataFrame(report_nb).T.round(3)
            st.dataframe(report_nb_df, use_container_width=True)

    with tab2:
        st.subheader("5-Fold Cross Validation Accuracy")
        fig, ax = plt.subplots(figsize=(7, 4))
        fold_labels = [f"Fold {i+1}" for i in range(len(cv_nb))]
        bars = ax.bar(fold_labels, cv_nb, color="#f39c12", edgecolor="white")
        ax.axhline(cv_nb.mean(), color="#e74c3c", linestyle="--",
                   label=f"Mean = {cv_nb.mean():.4f}")
        ax.set_ylim(0, 1.15)
        ax.set_title("5-Fold CV Accuracy — Naïve Bayes", fontweight="bold")
        ax.set_ylabel("Accuracy"); ax.legend()
        for bar, val in zip(bars, cv_nb):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PERBANDINGAN MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Perbandingan Model":
    st.header("🏆 Perbandingan Model")

    st.subheader("📋 Tabel Ringkasan Performa")
    summary_data = {
        "Model": ["K-Means Clustering", "Logistic Regression", "Naïve Bayes"],
        "Metrik Utama": ["Silhouette Score", "Test Accuracy", "Test Accuracy"],
        "Nilai": [f"{sil:.4f}", f"{acc_lr:.4f}", f"{acc_nb:.4f}"],
        "CV Mean (5-fold)": ["-", f"{cv_lr.mean():.4f}", f"{cv_nb.mean():.4f}"],
        "CV Std": ["-", f"± {cv_lr.std():.4f}", f"± {cv_nb.std():.4f}"],
    }
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("📊 Visualisasi Perbandingan Accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart accuracy
    models = ["Logistic\nRegression", "Naïve\nBayes"]
    accs   = [acc_lr, acc_nb]
    colors_bar = ["#3498db", "#f39c12"]
    bars = axes[0].bar(models, accs, color=colors_bar, edgecolor="white", width=0.4)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_title("Test Accuracy Perbandingan", fontweight="bold")
    axes[0].set_ylabel("Accuracy")
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")

    # CV comparison
    x = np.arange(5)
    width = 0.35
    axes[1].bar(x - width/2, cv_lr, width, label="Logistic Regression",
                color="#3498db", edgecolor="white")
    axes[1].bar(x + width/2, cv_nb, width, label="Naïve Bayes",
                color="#f39c12", edgecolor="white")
    axes[1].axhline(cv_lr.mean(), color="#3498db", linestyle="--", alpha=0.6)
    axes[1].axhline(cv_nb.mean(), color="#f39c12", linestyle="--", alpha=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Fold {i+1}" for i in range(5)])
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title("5-Fold CV Accuracy Comparison", fontweight="bold")
    axes[1].set_ylabel("Accuracy"); axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Winner
    winner = "Logistic Regression" if acc_lr >= acc_nb else "Naïve Bayes"
    st.success(f"✅ **Model Terbaik:** {winner} dengan Test Accuracy = {max(acc_lr, acc_nb):.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — PREDIKSI BARU
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediksi Baru":
    st.header("🔮 Prediksi Kualitas Susu")
    st.markdown("Masukkan nilai parameter susu untuk mendapatkan prediksi kualitas secara **real-time**.")

    col1, col2 = st.columns(2)
    with col1:
        ph_val   = st.slider("🧪 pH", min_value=3.0, max_value=9.5, value=6.6, step=0.1)
        temp_val = st.slider("🌡️ Temperatur (°C)", min_value=34, max_value=90, value=35)
        colour_val = st.slider("🎨 Colour (skala warna)", min_value=240, max_value=255, value=254)
    with col2:
        taste_val     = st.radio("👅 Taste", [0, 1], format_func=lambda x: "Buruk (0)" if x == 0 else "Baik (1)", horizontal=True)
        odor_val      = st.radio("👃 Odor", [0, 1], format_func=lambda x: "Tidak Berbau (0)" if x == 0 else "Berbau (1)", horizontal=True)
        fat_val       = st.radio("🧈 Fat", [0, 1], format_func=lambda x: "Rendah (0)" if x == 0 else "Tinggi (1)", horizontal=True)
        turbidity_val = st.radio("💧 Turbidity", [0, 1], format_func=lambda x: "Jernih (0)" if x == 0 else "Keruh (1)", horizontal=True)

    input_raw = pd.DataFrame([[ph_val, temp_val, taste_val, odor_val,
                                fat_val, turbidity_val, colour_val]],
                              columns=features)
    input_scaled = scaler.transform(input_raw)

    if st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True):
        pred_lr = lr.predict(input_scaled)[0]
        pred_nb = nb.predict(input_scaled)[0]
        prob_lr = lr.predict_proba(input_scaled)[0]
        prob_nb = nb.predict_proba(input_scaled)[0]

        grade_lr = le.inverse_transform([pred_lr])[0]
        grade_nb = le.inverse_transform([pred_nb])[0]

        col1, col2 = st.columns(2)
        grade_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}

        with col1:
            st.markdown(f"""
            <div style="background:#eaf4fb;border-left:5px solid #3498db;
                        padding:1rem;border-radius:8px;">
                <h3 style="color:#3498db;">📈 Logistic Regression</h3>
                <h2>{grade_emoji.get(grade_lr,'')} Grade: <b>{grade_lr.upper()}</b></h2>
            </div>""", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 2.5))
            cls_names = le.classes_
            ax.barh(cls_names, prob_lr, color=["#2ecc71","#e74c3c","#f39c12"], edgecolor="black")
            ax.set_xlim(0, 1); ax.set_xlabel("Probabilitas")
            ax.set_title("Probabilitas per Kelas — LR", fontweight="bold")
            for i, v in enumerate(prob_lr):
                ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col2:
            st.markdown(f"""
            <div style="background:#fef9e7;border-left:5px solid #f39c12;
                        padding:1rem;border-radius:8px;">
                <h3 style="color:#f39c12;">🟡 Naïve Bayes</h3>
                <h2>{grade_emoji.get(grade_nb,'')} Grade: <b>{grade_nb.upper()}</b></h2>
            </div>""", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.barh(cls_names, prob_nb, color=["#2ecc71","#e74c3c","#f39c12"], edgecolor="black")
            ax.set_xlim(0, 1); ax.set_xlabel("Probabilitas")
            ax.set_title("Probabilitas per Kelas — NB", fontweight="bold")
            for i, v in enumerate(prob_nb):
                ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        if grade_lr == grade_nb:
            st.success(f"✅ Kedua model **setuju**: Kualitas susu **{grade_lr.upper()}**")
        else:
            st.warning(f"⚠️ Hasil berbeda — LR: **{grade_lr.upper()}** | NB: **{grade_nb.upper()}**")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;font-size:0.85rem;'>"
    "🥛 Milk Quality Dashboard · BBK2LAB3 Penambangan Data · Telkom University 2025/2026"
    "</p>",
    unsafe_allow_html=True
)
