# ML Model Playground 🧠

An interactive, reproducible machine-learning workbench built with **Streamlit** and **scikit-learn**.

### Features
- Compare Logistic Regression, Random Forest and K-Nearest Neighbors
- Explore Iris, Wine and Breast Cancer datasets
- Train/test split with stratification and reproducible seed
- Accuracy, macro-F1, classification report and confusion matrix
- Download predictions as CSV
- No API keys, paid services, or external dataset downloads required

## Quick start
```bash
git clone https://github.com/iamAdarshAnand/ML-Model-Playground.git
cd ML-Model-Playground
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Testing
```bash
pytest -q
```

## Project structure
```
app.py             Streamlit UI
ml_core.py         Dataset loading, training and evaluation
tests/             Automated smoke tests
requirements.txt   Runtime and test dependencies
```

## Notes
The datasets are bundled with scikit-learn. Metrics describe held-out test data, not real-world deployment performance. Breast Cancer data is used for education only and is **not** a diagnostic tool.

## License
MIT. Created by Adarsh Anand.

## Project discovery

**Interactive Machine Learning Model Comparison** — Streamlit scikit-learn dashboard for Logistic Regression, Random Forest and KNN model evaluation using public educational datasets.

**Related developer searches:** machine learning playground, scikit-learn, Streamlit, model comparison, classification, confusion matrix, Python.

**Source:** [ML-Model-Playground](https://github.com/iamAdarshAnand/ML-Model-Playground) · **Author:** [Adarsh Anand](https://github.com/iamAdarshAnand)
