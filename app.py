# ============================================================
# Streamlit Web App — Student Performance Prediction
# Run : streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
)

# ------------------------------------------------------------------
# Load trained model (train it on-the-fly if .pkl not found)
# ------------------------------------------------------------------
MODEL_PATH = "linear_regression_model.pkl"

@st.cache_resource
def load_model():
    """Load the pre-trained model or train a fresh one."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    # Fallback: train on the fly
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("Student_Performance.csv")
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map(
        {"Yes": 1, "No": 0}
    )
    FEATURES = [
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced",
    ]
    X = df[FEATURES]
    y = df["Performance Index"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=42)
    m = LinearRegression()
    m.fit(X_train, y_train)
    joblib.dump(m, MODEL_PATH)
    return m


model = load_model()

FEATURES = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🎓 Student Performance Predictor")
st.markdown(
    """
    Enter the five study-related details below and click **Predict** to get
    an estimated **Performance Index** (scale 0 – 100).
    """
)
st.divider()

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    hours_studied = st.slider(
        "📚 Hours Studied (per day)",
        min_value=1, max_value=9, value=5, step=1,
        help="Average number of hours the student studies per day.",
    )
    previous_scores = st.slider(
        "📝 Previous Scores",
        min_value=40, max_value=99, value=70, step=1,
        help="Score obtained in the previous exam (out of 100).",
    )
    extracurricular = st.radio(
        "🏅 Extracurricular Activities",
        options=["Yes", "No"],
        index=0,
        horizontal=True,
        help="Does the student participate in extracurricular activities?",
    )

with col2:
    sleep_hours = st.slider(
        "😴 Sleep Hours (per day)",
        min_value=4, max_value=9, value=7, step=1,
        help="Average number of hours of sleep per day.",
    )
    papers_practiced = st.slider(
        "📄 Sample Papers Practiced",
        min_value=0, max_value=9, value=3, step=1,
        help="Number of sample question papers practiced.",
    )

st.divider()

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if st.button("🔮 Predict Performance Index", use_container_width=True):
    ext_encoded = 1 if extracurricular == "Yes" else 0

    input_df = pd.DataFrame(
        [[hours_studied, previous_scores, ext_encoded, sleep_hours, papers_practiced]],
        columns=FEATURES,
    )

    prediction = model.predict(input_df)[0]
    # Clip to valid range
    prediction = float(np.clip(prediction, 0, 100))

    # Colour feedback
    if prediction >= 75:
        colour = "green"
        level = "Excellent 🏆"
    elif prediction >= 55:
        colour = "orange"
        level = "Good 👍"
    else:
        colour = "red"
        level = "Needs Improvement ⚠️"

    st.markdown(
        f"""
        <div style="
            background:#f7f8fa;
            border-left:6px solid {colour};
            padding:20px 24px;
            border-radius:6px;
            margin-top:8px;">
            <h2 style="margin:0;color:{colour};">
                Predicted Performance Index: {prediction:.2f}
            </h2>
            <p style="margin:6px 0 0;font-size:16px;color:#444;">
                Performance Level: <strong>{level}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Mini bar chart of input features -------------------------
    st.subheader("📊 Your Input at a Glance")
    fig, ax = plt.subplots(figsize=(7, 3))
    labels = ["Hours\nStudied", "Previous\nScores", "Extracurricular\n(1=Yes)", "Sleep\nHours", "Papers\nPracticed"]
    values = [hours_studied, previous_scores, ext_encoded, sleep_hours, papers_practiced]
    bars = ax.barh(labels, values, color=["#3b82d4", "#7c5cd8", "#22c55e", "#f59e0b", "#ef4444"])
    ax.set_xlabel("Value")
    ax.set_title("Input Feature Values")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10)
    ax.set_xlim(0, max(values) * 1.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ------------------------------------------------------------------
# Sidebar: Model info & graphs
# ------------------------------------------------------------------
st.sidebar.header("ℹ️ Model Information")
st.sidebar.markdown(
    """
    - **Algorithm:** Linear Regression
    - **Dataset:** Student_Performance.csv
    - **Rows:** 10,000
    - **Train/Test Split:** 80 / 20
    - **Target:** Performance Index (0–100)
    """
)

st.sidebar.header("📈 Dataset Graphs")
output_graphs = {
    "Hours Studied vs Performance": "outputs/hours_vs_performance.png",
    "Previous Scores vs Performance": "outputs/scores_vs_performance.png",
    "Actual vs Predicted": "outputs/actual_vs_predicted.png",
    "Correlation Heatmap": "outputs/correlation_heatmap.png",
}
for label, path in output_graphs.items():
    if os.path.exists(path):
        st.sidebar.image(path, caption=label, use_container_width=True)
    else:
        st.sidebar.caption(f"Run student_performance.py to generate: {label}")

st.sidebar.divider()
st.sidebar.caption("Built with Python · scikit-learn · Streamlit")
