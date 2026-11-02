# Streamlit app for Starbucks Review Rating Prediction
# Keep this file in the same folder as:
#   1. archive (2).zip
#   2. predictions_comparison.csv   (optional)
#   3. starbucks_data.py            (original notebook-export script; not required to run this app)
#
# Install dependencies:
#   pip install streamlit pandas scikit-learn matplotlib

import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "archive (2).zip"
COMPARISON_PATH = BASE_DIR / "predictions_comparison.csv"


st.set_page_config(
    page_title="Starbucks Review Rating Predictor",
    page_icon="☕",
    layout="wide"
)


@st.cache_data
def load_dataset(zip_file_path: str) -> pd.DataFrame:
    """Read reviews_data.csv directly from the supplied ZIP file."""
    with zipfile.ZipFile(zip_file_path, "r") as archive:
        csv_names = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]
        if not csv_names:
            raise FileNotFoundError("ZIP file ke andar koi CSV dataset nahi mila.")
        with archive.open(csv_names[0]) as csv_file:
            data = pd.read_csv(csv_file)
    return data


@st.cache_resource
def train_model(data: pd.DataFrame):
    """Train the same TF-IDF + Decision Tree approach as the provided script."""
    required = {"Review", "Rating"}
    if not required.issubset(data.columns):
        raise ValueError("Dataset mein 'Review' aur 'Rating' columns required hain.")

    model_data = data.dropna(subset=["Review", "Rating"]).copy()
    model_data["Review"] = model_data["Review"].astype(str).str.strip()
    model_data = model_data[model_data["Review"] != ""]
    model_data["Rating"] = pd.to_numeric(model_data["Rating"], errors="coerce")
    model_data = model_data.dropna(subset=["Rating"])
    model_data["Rating"] = model_data["Rating"].astype(int)

    # Keep only rating classes with at least two examples so stratified split works.
    counts = model_data["Rating"].value_counts()
    model_data = model_data[model_data["Rating"].isin(counts[counts >= 2].index)]

    if model_data["Rating"].nunique() < 2:
        raise ValueError("Model train karne ke liye kam az kam 2 rating classes chahiye.")

    X_train, X_test, y_train, y_test = train_test_split(
        model_data["Review"],
        model_data["Rating"],
        test_size=0.2,
        random_state=42,
        stratify=model_data["Rating"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english")),
        ("classifier", DecisionTreeClassifier(max_depth=15, random_state=42))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions)

    comparison = pd.DataFrame({
        "Review Text": X_test.reset_index(drop=True),
        "Actual Rating": y_test.reset_index(drop=True),
        "Predicted Rating": predictions
    })
    return model, accuracy, report, matrix, comparison, model_data


st.title("☕ Starbucks Review Rating Prediction")
st.write(
    "Customer review enter karein. App review text ko analyze karke "
    "predicted Starbucks rating show karegi."
)

if not ZIP_PATH.exists():
    st.error(
        "Dataset ZIP file nahi mili. `archive (2).zip` ko app.py ke same folder "
        "mein rakhein."
    )
    st.stop()

try:
    dataset = load_dataset(str(ZIP_PATH))
    model, accuracy, report, matrix, comparison, clean_data = train_model(dataset)
except Exception as error:
    st.error(f"App/model load nahi ho saka: {error}")
    st.stop()

# Sidebar information
st.sidebar.header("Dataset Information")
st.sidebar.metric("Total dataset rows", f"{len(dataset):,}")
st.sidebar.metric("Usable reviews", f"{len(clean_data):,}")
st.sidebar.metric("Model test accuracy", f"{accuracy:.2%}")
st.sidebar.caption("Algorithm: TF-IDF + Decision Tree (max_depth=15)")

tab_predict, tab_analysis, tab_data = st.tabs(
    ["Predict Rating", "Model Performance", "Dataset"]
)

with tab_predict:
    st.subheader("Enter a customer review")
    review_text = st.text_area(
        "Review",
        placeholder="Example: The staff was friendly and my coffee was excellent...",
        height=150
    )
    if st.button("Predict Rating", type="primary"):
        if not review_text.strip():
            st.warning("Please pehle review text enter karein.")
        else:
            prediction = int(model.predict([review_text.strip()])[0])
            st.success(f"Predicted Rating: {prediction} / 5")
            st.caption(
                "Prediction model ke learned patterns par based hai; "
                "yeh guaranteed customer rating nahi hai."
            )

with tab_analysis:
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Training reviews", f"{int(len(clean_data) * 0.8):,}")
    col3.metric("Testing reviews", f"{len(comparison):,}")

    st.markdown("**Classification report**")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

    st.markdown("**Confusion matrix**")
    labels = sorted(clean_data["Rating"].unique())
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels]
    )
    st.dataframe(matrix_df, use_container_width=True)

    st.markdown("**Actual vs Predicted (test samples)**")
    st.dataframe(comparison.head(20), use_container_width=True)
    csv_bytes = comparison.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download test predictions CSV",
        data=csv_bytes,
        file_name="predictions_comparison.csv",
        mime="text/csv"
    )

with tab_data:
    st.subheader("Dataset Preview")
    st.write(f"Dataset shape: {dataset.shape[0]} rows × {dataset.shape[1]} columns")
    st.dataframe(dataset.head(30), use_container_width=True)

    if "Rating" in dataset.columns:
        st.markdown("**Rating distribution**")
        rating_counts = dataset["Rating"].value_counts().sort_index()
        st.bar_chart(rating_counts)

    if COMPARISON_PATH.exists():
        st.markdown("**Previously saved prediction comparison file**")
        try:
            old_comparison = pd.read_csv(COMPARISON_PATH)
            st.dataframe(old_comparison.head(20), use_container_width=True)
        except Exception as error:
            st.warning(f"Comparison CSV read nahi ho saki: {error}")

st.caption("Starbucks Review Rating Prediction • Streamlit")
