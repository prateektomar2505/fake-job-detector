
import streamlit as st

st.markdown('''
<style>
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
.percentage-text {
    color: #00FF00 !important;
    -webkit-text-fill-color: #00FF00 !important;
    font-weight: 900 !important;
    opacity: 1 !important;
}
</style>
''', unsafe_allow_html=True)


import streamlit as st
import joblib
import os
import pandas as pd

st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
h1 {
    color: #17365D;
}
h2, h3 {
    color: #24476B;
}
div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #dce3ed;
    padding: 18px;
    border-radius: 12px;
}
.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 3em;
    background-color: #17365D;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.title("🔍 Fake Job Posting Detection")
st.write(
    "An AI-powered application to identify potentially fraudulent job postings."
)

st.info(
    "Enter the job details below to estimate whether the posting "
    "may be fraudulent."
)

# ---------------- LOAD MODEL ----------------

MODEL_PATH = "models/job_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
METRICS_PATH = "models/performance_metrics.pkl"
CM_PATH = "models/confusion_matrix.png"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found: " + MODEL_PATH)
    st.stop()

if not os.path.exists(VECTORIZER_PATH):
    st.error("TF-IDF vectorizer not found.")
    st.stop()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ---------------- JOB INPUT ----------------

st.header("📝 Enter Job Details")

col1, col2 = st.columns(2)

with col1:
    title = st.text_input(
        "Job Title",
        placeholder="e.g. Data Entry Operator"
    )

    company = st.text_area(
        "Company Profile",
        placeholder="Enter company information..."
    )

with col2:
    location = st.text_input(
        "Job Location (Optional)",
        placeholder="e.g. New Delhi, India"
    )

    department = st.text_input(
        "Department (Optional)",
        placeholder="e.g. Operations"
    )

description = st.text_area(
    "Job Description",
    placeholder="Enter the complete job description...",
    height=180
)

# ---------------- PREDICTION ----------------

if st.button("🔍 Analyze Job Posting"):

    if not title.strip() and not description.strip():
        st.warning("Please enter a job title or job description.")

    else:
        text = " ".join([
            title,
            company,
            description,
            location,
            department
        ])

        vectorized_text = vectorizer.transform([text])
        prediction = model.predict(vectorized_text)[0]

        st.divider()
        st.header("📋 Prediction Result")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vectorized_text)[0]
            classes = list(model.classes_)

            if 1 in classes:
                fake_probability = float(
                    probabilities[classes.index(1)]
                )

                fake_percent = fake_probability * 100
                real_percent = (1 - fake_probability) * 100

                if prediction == 1:
                    st.error("⚠️ This job posting may be FAKE.")
                else:
                    st.success("✅ This job posting appears REAL.")

                result1, result2 = st.columns(2)

                with result1:
                    st.metric(
                        "Estimated Fake Probability",
                        f"{fake_percent:.2f}%"
                    )
                    st.progress(fake_probability)

                with result2:
                    st.metric(
                        "Estimated Real Probability",
                        f"{real_percent:.2f}%"
                    )
                    st.progress(1 - fake_probability)

                st.caption(
                    "These percentages are model-estimated probabilities, "
                    "not a guarantee that the job is genuine or fraudulent."
                )

            else:
                st.write("Prediction:", prediction)

        else:
            if prediction == 1:
                st.error("⚠️ This job posting may be FAKE.")
            else:
                st.success("✅ This job posting appears REAL.")

            st.warning("This model does not support probability estimates.")

# ---------------- MODEL PERFORMANCE ----------------

st.divider()
st.header("📊 Model Performance Dashboard")

if os.path.exists(METRICS_PATH):

    metrics = joblib.load(METRICS_PATH)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.2f}%")

    with m2:
        st.metric("Precision", f"{metrics['Precision']:.2f}%")

    with m3:
        st.metric("Recall", f"{metrics['Recall']:.2f}%")

    with m4:
        st.metric("F1 Score", f"{metrics['F1 Score']:.2f}%")

else:
    st.warning("Performance metrics file not found.")

if os.path.exists(CM_PATH):
    st.subheader("Confusion Matrix")
    st.image(
        CM_PATH,
        caption="Model evaluation on the test split",
        use_container_width=True
    )

st.divider()

st.caption(
    "Fake Job Posting Detection | Machine Learning Project | "
    "Predictions should be verified independently."
)
