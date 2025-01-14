from sklearn.model_selection import train_test_split
import mlflow.spacy
from typing import List
import pandas as pd

# Constantes
SEED = 314

# Fonctions - Data loading, splitting and preprocessing
def show_nums_axes(ax, orient='v', fmt='.0g', stacked=False):
    """
    Affiche les valeurs numériques sur les barres d'un graphique à barres.

    Args:
        ax (matplotlib.axes.Axes): L'axe du graphique.
        fmt (str, optional): Format d'affichage des nombres
        orient (str, optional): L'orientation des barres. 'v' pour vertical (par défaut), 'h' pour horizontal.
        stacked (bool, optionnal): S'adapte pour un barstackedplot

    Returns:
        None
    """
    # Error management
    if orient not in ['h', 'v']:
        raise ValueError("orient doit être égal à 'h' ou 'v si spécifié")
    try:
        format(-10.5560, fmt)
    except ValueError as e:
        raise f"{e}: le format spécifié dans fmt n'est pas correct."
    if not isinstance(stacked, bool):
        raise ValueError("stacked doit être un booléen")
    # Body
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if orient == 'v':
            if not stacked:
                ax.annotate(f'{height:{fmt}}' if height!=0 else '',
                            (x + width / 2., height), ha='center', va='bottom')
            else:
                ax.annotate(f'{height:{fmt}}', (x + width-4, y + height/2),
                            fontsize=10, fontweight='bold', ha='center', va='top')
        else:
            if not stacked:
                ax.annotate(f'{width:{fmt}}' if width!=0 else '',
                            (width, y + height / 2.), ha='left', va='center')
            else:
                ax.annotate(f'{width:{fmt}}', (x + width-1, y + height/2),
                            fontsize=10, fontweight='bold', ha='right', va='center')

# Fonctions - MLflow
def get_mlflow_results(experiment_name, uri="http://localhost:5000"):
    client = mlflow.tracking.MlflowClient(uri)
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    # Préparer une liste de dictionnaires pour chaque run
    runs_data = []

    for run in runs:
        # Extraire des informations utiles pour chaque run
        run_info = run.info
        run_data = {
            "run_id": run_info.run_id,
            "status": run_info.status,
            "start_time": run_info.start_time,
            "end_time": run_info.end_time,
            "artifact_uri": run_info.artifact_uri,
        }

        # Inclure paramètres
        for param_key, param_value in run.data.params.items():
            run_data[f"param_{param_key}"] = param_value

        # Inclure métriques
        for metric_key, metric_value in run.data.metrics.items():
            run_data[f"metric_{metric_key}"] = metric_value

        # Inclure tags
        for tag_key, tag_value in run.data.tags.items():
            run_data[f"tag_{tag_key}"] = tag_value

        runs_data.append(run_data)

    # Créer un DataFrame à partir de la liste de dictionnaires
    return pd.DataFrame(runs_data)

# Fonctions - Data loading, splitting and preprocessing
def split_data(df, test_split=0.2, sampling=True, proportion=0.01, stratify=True):
    """
    Split the data into train and test sets
    :param test_split: Proportion of the data to include in the test split
    :param sampling: Whether to sample the data
    :param proportion: Proportion of the data to sample
    :return: X_train, X_test, y_train, y_test
    """
    assert df.columns[0] == "target", "The first column must be the target column"
    assert len(df.columns) > 1, "At least one feature column must be provided"
    # Sample the data if needed to reduce the size
    if sampling:
        df_sample = df.sample(frac=proportion, random_state=SEED)
    else:
        df_sample = df
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df_sample.iloc[:, 1:],
        df_sample.iloc[:, 0],
        test_size=test_split,
        stratify=df_sample.iloc[:, 0] if stratify else None,
        random_state=SEED,
    )
    # Delete the sample dataframe to free up memory
    del df_sample
    # Return the train and test sets
    return X_train.squeeze(), X_test.squeeze(), y_train, y_test

def filter_dataset(X_train, X_test, cols):
    """
    Filter the columns of the train and test sets
    """
    # Define columns to read
    assert isinstance(cols, list)
    assert set(cols).issubset(set(X_train.columns)), f"Some column in {cols} not found in the dataframe"
    assert set(cols).issubset(set(X_test.columns)), f"Some column in {cols} not found in the dataframe"

    # Align the dataframes and reindex in the same order
    X_train = X_train.filter(cols)
    X_test = X_test.filter(cols)

    # Return the aligned data with features squeezed to remove the extra dimension if necessary
    return X_train, X_test


class SpacyTokenizer:
    def __init__(self, model_uri: str):
        self.nlp = mlflow.spacy.load_model(model_uri)

    def tokenize(self, texts: List[str], lemmatize=False):
        # Error handling for input
        assert isinstance(texts, list), "Input must be a list of strings"
        assert isinstance(lemmatize, bool), "lemmatize must be a boolean"
        assert isinstance(texts[0], str), "List elements must be strings"
        # Tokenize the texts
        tokenized_texts = []
        # Tokenize and lemmatize
        if lemmatize:
            for doc in self.nlp.pipe(texts, disable=["ner"]):
                tokenized_texts.append(" ".join([token.lemma_ for token in doc]))
        # Tokenize the text only
        else:
            for doc in self.nlp.pipe(texts, disable=["ner", "lemmatizer"]):
                tokenized_texts.append(" ".join([token.text for token in doc]))
        # Return the tokenized texts
        return tokenized_texts

# Définition de la fonction de preprocessing
def text_preprocessing(serie: pd.Series, remove_punct_and_digits=False) -> pd.Series:
    # Clean but keep the punctuation
    if not remove_punct_and_digits:
        serie = (
            serie.str.lower()  # lowercase the text
            .str.replace(r"https?://?\S+", " ", regex=True)  # remove urls
            .str.replace(r"www\.\w+\S+", " ", regex=True)  # remove urls
            .str.replace(r"@\w+", "@usertag", regex=True)  # remove mentions
            .str.replace(r"#\w+", " ", regex=True)  # remove hashtags
            .str.replace(r"(\w+)[.,!?;](\w+)", r"\1 \2", regex=True)
            .str.replace(r"\s{2,}", " ", regex=True)  # strip multiple spaces
            .str.strip()
        )
        return serie
    # Cleaned and remove all punctuation and numbers
    else:
        serie = (
            serie.str.lower()  # lowercase the text
            .str.replace(r"https?://?\S+", " ", regex=True)  # remove urls
            .str.replace(r"www\.\w+\S+", " ", regex=True)  # remove urls
            .str.replace(r"@\w+", "@usertag", regex=True)  # remove mentions
            .str.replace(r"#\w+", " ", regex=True)  # remove hashtags
            .str.replace(r"(\w+)[.,!?;](\w+)", r"\1 \2", regex=True)
            .str.replace(
                r"[\\/.,!?;_:()=<>\[\]\-]", " ", regex=True
            )  # replace special chars and punct '
            .str.replace(r"[0-9]", "", regex=True)  # remove numbers
            .str.replace(r"\s{2,}", " ", regex=True)  # strip multiple spaces
            .str.strip()
        )
        return serie