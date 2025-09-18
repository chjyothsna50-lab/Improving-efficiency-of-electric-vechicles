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
        <h3 style="color:#117A65;">Manual Data Entry & Analysis</h3>
        <p><b>👩‍💻 Made by Jyothsna</b> during internship at <b>Edunet x Shell</b></p>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("---")

# -------------------------------
# ✍️ Data Entry Form
# -------------------------------
st.sidebar.header("✍️ Enter EV Data")

if "ev_data" not in st.session_state:
    st.session_state.ev_data = pd.DataFrame(columns=["Model", "Efficiency_kmPerkWh"])

with st.sidebar.form("data_form"):
    model_name = st.text_input("Car Model")
    efficiency = st.number_input("Efficiency (km/kWh)", min_value=0.0, step=0.1)
    submitted = st.form_submit_button("Add Entry")
    if submitted and model_name:
        st.session_state.ev_data = pd.concat(
            [st.session_state.ev_data, pd.DataFrame({"Model": [model_name], "Efficiency_kmPerkWh": [efficiency]})],
            ignore_index=True
        )
        st.success(f"✅ Added {model_name} with {efficiency} km/kWh")

# -------------------------------
# 📊 Show Data
# -------------------------------
df = st.session_state.ev_data

if df.empty:
    st.info("👉 Enter EV details from the sidebar to start analysis.")
else:
    st.subheader("📋 Entered EV Data")
    st.dataframe(df, use_container_width=True)

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
    ax.hist(df["Efficiency_kmPerkWh"], bins=5, color="#2E86C1", edgecolor="white")
    ax.set_xlabel("Efficiency (km/kWh)")
    ax.set_ylabel("Number of Models")
    st.pyplot(fig)

    # -------------------------------
    # 💾 Download Data
    # -------------------------------
    st.subheader("💾 Download Data")
    st.download_button(
        label="Download Entered Data (CSV)",
        data=df.to_csv(index=False),
        file_name="ev_efficiency_data.csv",
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
