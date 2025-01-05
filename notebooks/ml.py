from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_ml_model(text_col, vectorizer, classifier):
    """
    Creates a machine learning model pipeline with a specified text vectorizer and classifier.

    Parameters:
    text_col (str): The name of the column containing text data to be vectorized.
    vectorizer (sklearn.base.TransformerMixin): The vectorizer to convert text data into numerical features (e.g., TfidfVectorizer).
    classifier (sklearn.base.ClassifierMixin): The classifier to be used for prediction (e.g., LogisticRegression).

    Returns:
    sklearn.pipeline.Pipeline: A scikit-learn pipeline object that preprocesses the text data and applies the classifier.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "vectorizer",
                vectorizer,
                text_col,
            ),
        ],
        remainder="passthrough",
    )
    # create a pipeline with Tf-Idf and Logistic Regression
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return model