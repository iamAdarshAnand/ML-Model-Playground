"""Streamlit interface for the ML Model Playground."""
import pandas as pd
import streamlit as st
from ml_core import DATASETS, MODELS, load_dataset, train_evaluate

st.set_page_config(page_title="ML Model Playground", page_icon="🧠", layout="wide")
st.title("🧠 ML Model Playground")
st.caption("Explore datasets, train classifiers, and understand evaluation metrics.")

with st.sidebar:
    st.header("Experiment settings")
    dataset_name = st.selectbox("Dataset", list(DATASETS))
    model_name = st.selectbox("Classifier", MODELS)
    test_size = st.slider("Test fraction", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42)

dataset = load_dataset(dataset_name)
st.subheader("Dataset overview")
a, b, c = st.columns(3)
a.metric("Samples", len(dataset.data))
b.metric("Features", dataset.data.shape[1])
c.metric("Classes", len(dataset.target_names))
st.dataframe(dataset.frame.head(12), use_container_width=True)
st.bar_chart(dataset.target.value_counts().sort_index().rename(index=dict(enumerate(dataset.target_names))))

if st.button("🚀 Train and evaluate", type="primary"):
    with st.spinner("Training..."):
        result = train_evaluate(dataset_name, model_name, test_size, int(seed))
    st.subheader("Evaluation")
    left, right = st.columns(2)
    left.metric("Accuracy", f"{result.accuracy:.2%}")
    right.metric("Macro F1", f"{result.macro_f1:.3f}")
    st.write("Confusion matrix (rows: actual, columns: predicted)")
    st.dataframe(pd.DataFrame(result.confusion, index=result.labels, columns=result.labels), use_container_width=True)
    st.code(result.report, language="text")
    predictions = pd.DataFrame({
        "actual": [result.labels[int(v)] for v in result.truth],
        "predicted": [result.labels[int(v)] for v in result.predictions],
    })
    st.download_button("Download predictions CSV", predictions.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
st.caption("Educational demonstration. Results are not validated for production or clinical use.")
