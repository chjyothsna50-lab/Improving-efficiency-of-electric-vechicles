import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 🎨 Page Config
# -------------------------------
st.set_page_config(
    page_title="Energy Efficiency of Electric Vehicles",
    page_icon="🚗",
    layout="wide"
)

# -------------------------------
# 🏷️ Header Section
# -------------------------------
st.markdown(
    """
    <div style="text-align:center">
        <h1 style="color:#2E86C1;">🚗 Energy Efficiency of Electric Vehicles</h1>
        <h3 style="color:#117A65;">Analyzing and Visualizing EV Efficiency</h3>
        <p><b>👩‍💻 Made by Jyothsna</b> during internship at <b>Edunet x Shell</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# -------------------------------
# 📂 File Upload Section
# -------------------------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

def load_data(file):
    if file is None:
        return None
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    return None

df = load_data(uploaded_file)

# -------------------------------
# 📊 Main Content
# -------------------------------
if df is None:
    st.info("👆 Upload a dataset from the sidebar to begin analysis.")
else:
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10))

    # Efficiency column check
    if "Efficiency_kmPerkWh" not in df.columns:
        st.error("Dataset must have a column named `Efficiency_kmPerkWh`.")
    else:
        # -------------------------------
        # 🔑 Key Metrics
        # -------------------------------
        avg_eff = round(df["Efficiency_kmPerkWh"].mean(), 2)
        best_model = df.loc[df["Efficiency_kmPerkWh"].idxmax()]
        worst_model = df.loc[df["Efficiency_kmPerkWh"].idxmin()]

        col1, col2, col3 = st.columns(3)
        col1.metric("⚡ Average Efficiency", f"{avg_eff} km/kWh")
        col2.metric("🥇 Best Model", f"{best_model['Model']} ({round(best_model['Efficiency_kmPerkWh'],2)})")
        col3.metric("🥲 Worst Model", f"{worst_model['Model']} ({round(worst_model['Efficiency_kmPerkWh'],2)})")

        st.write("---")

        # -------------------------------
        # 📊 Visualizations
        # -------------------------------
        st.subheader("📊 Efficiency by Model")
        st.bar_chart(df.set_index("Model")["Efficiency_kmPerkWh"])

        st.subheader("📈 Distribution of Efficiency")
        fig, ax = plt.subplots()
        ax.hist(df["Efficiency_kmPerkWh"], bins=10, color="#2E86C1", edgecolor="white")
        ax.set_xlabel("Efficiency (km/kWh)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Correlation heatmap if numeric cols exist
        numeric_cols = df.select_dtypes(include=[np.number])
        if len(numeric_cols.columns) > 1:
            st.subheader("📊 Correlation Heatmap")
            corr = numeric_cols.corr()

            fig, ax = plt.subplots(figsize=(6, 4))
            cax = ax.matshow(corr, cmap="coolwarm")
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
            plt.yticks(range(len(corr.columns)), corr.columns)
            fig.colorbar(cax)
            st.pyplot(fig)

        # -------------------------------
        # 💾 Download Processed Data
        # -------------------------------
        st.subheader("💾 Download Data")
        st.download_button(
            label="Download Cleaned CSV",
            data=df.to_csv(index=False),
            file_name="ev_efficiency_cleaned.csv",
            mime="text/csv"
        )

# -------------------------------
# 📌 Footer
# -------------------------------
st.write("---")
st.markdown(
    """
    <div style="text-align:center; color:grey;">
        🚀 Built with Streamlit | Internship Project by <b>Jyothsna</b>
    </div>
    """,
    unsafe_allow_html=True
)
