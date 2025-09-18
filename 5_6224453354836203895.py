
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="EV Efficiency Analysis (Explorer)", layout="wide")

st.title("EV Efficiency Analysis — Interactive Explorer")
st.markdown(
    """
Upload your dataset (CSV or Excel). This lightweight app focuses on **data loading, exploration, and visualization** only — no model training.
- Supports CSV, XLS, XLSX.
- Shows raw data, summary statistics, basic plots (histograms, scatter, correlation heatmap), and missing-data overview.
"""
)

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])

def load_dataframe(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return None
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None
    return df

df = load_dataframe(uploaded_file)

if df is None:
    st.info("Upload a dataset to begin. You can also drag-and-drop a CSV or Excel file.")
else:
    st.header("Preview & Basic Info")
    st.write(f"**Filename:** {uploaded_file.name}  —  **Rows:** {df.shape[0]}  **Columns:** {df.shape[1]}")
    if st.checkbox("Show raw data (first 100 rows)"):
        st.dataframe(df.head(100))

    st.subheader("Column types and missing values")
    col_info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "num_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2)
    })
    st.table(col_info)

    st.subheader("Summary statistics (numeric)")
    st.dataframe(df.select_dtypes(include=[np.number]).describe().T)

    st.subheader("Missing-data heatmap (simple)")
    fig, ax = plt.subplots(figsize=(8, min(6, 0.2*df.shape[1] + 2)))
    ax.imshow(df.isna().T, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)
    ax.set_xlabel("Row index")
    ax.set_title("Missing values (black = missing)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        st.warning("No numeric columns found — plotting and statistics are limited.")
    else:
        st.subheader("Histograms")
        selected_hist = st.multiselect("Choose numeric columns to histogram", numeric_cols, default=numeric_cols[:3])
        for col in selected_hist:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.hist(df[col].dropna(), bins=30)
            ax.set_title(f"Histogram — {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Scatter plot")
        x_col = st.selectbox("Select X axis (numeric)", numeric_cols, index=0)
        y_col = st.selectbox("Select Y axis (numeric)", numeric_cols, index=min(1, len(numeric_cols)-1))
        sample_frac = st.slider("Sample fraction for scatter (to keep plots fast)", 0.01, 1.0, 0.2, step=0.01)
        plot_df = df[[x_col, y_col]].dropna()
        if sample_frac < 1.0:
            plot_df = plot_df.sample(frac=sample_frac, random_state=42)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(plot_df[x_col], plot_df[y_col], alpha=0.6, s=10)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Correlation matrix (numeric columns)")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(max(6, 0.6*len(numeric_cols)), max(4, 0.4*len(numeric_cols))))
        cax = ax.imshow(corr.values, interpolation="nearest", aspect="auto")
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=90)
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_yticklabels(numeric_cols)
        ax.set_title("Correlation matrix")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Download cleaned sample (first 100 rows)")
    buf = BytesIO()
    df.head(100).to_csv(buf, index=False)
    st.download_button("Download sample CSV", data=buf.getvalue(), file_name="ev_sample.csv", mime="text/csv")

    st.info("App created for exploration only. If you'd like model evaluation or specific plots from your notebook ported, reply 'include model' or list the exact plots to add.")
