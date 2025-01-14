import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
import pandas as pd
import os

# Constantes
SEED = 314

@tf.keras.utils.register_keras_serializable(package="custom_text_func", name="custom_standardization_punct")
def custom_standardization_punct(tensor):
    tensor = tf.strings.lower(tensor)  # lowercase
    tensor = tf.strings.regex_replace(tensor, r"https?://?\S+", " ")  # strip mentions
    tensor = tf.strings.regex_replace(tensor, r"www\.\w+\S+", " ")  # strip mentions
    tensor = tf.strings.regex_replace(tensor, r"@\w+", "@usertag")
    tensor = tf.strings.regex_replace(tensor, r"#\w+", " ")
    tensor = tf.strings.regex_replace(tensor, r"(\w+)[.,!?;](\w+)", r"\1 \2")
    tensor = tf.strings.regex_replace(tensor, r"\s{2,}", " ")  # strip multiple spaces
    return tf.strings.strip(tensor)  # strip leading and trailing spaces

@tf.keras.utils.register_keras_serializable(package="custom_text_func", name="custom_standardization_nopunct_digits")
def custom_standardization_nopunct_digits(tensor):
    tensor = tf.strings.lower(tensor)  # lowercase
    tensor = tf.strings.regex_replace(tensor, r"https?://?\S+", " ")  # strip mentions
    tensor = tf.strings.regex_replace(tensor, r"www\.\w+\S+", " ")  # strip mentions
    tensor = tf.strings.regex_replace(tensor, r"@\w+", "@usertag")
    tensor = tf.strings.regex_replace(tensor, r"#\w+", " ")
    tensor = tf.strings.regex_replace(tensor, r"(\w+)[.,!?;](\w+)", r"\1 \2")
    tensor = tf.strings.regex_replace(tensor, r"[\\/.,!?;_:()=<>\[\]\-]", " ")  # replace special chars and punct
    tensor = tf.strings.regex_replace(tensor, r"[0-9]", " ")  # remove numbers
    tensor = tf.strings.regex_replace(tensor, r"\s{2,}", " ")  # strip multiple spaces
    return tf.strings.strip(tensor)  # strip leading and trailing spaces


@tf.keras.utils.register_keras_serializable(package="custom_layer", name="TextVectorizer")
class TextVectorizer(tf.keras.layers.Layer):
    def __init__(self, max_tokens=None, output_mode='int', output_sequence_length=None, standardize_func=None, trainable=False, vocabulary=None):
        super().__init__()
        self.max_tokens = max_tokens
        self.output_mode = output_mode
        self.output_sequence_length = output_sequence_length
        self.standardize = standardize_func
        self.vocabulary = vocabulary
        self.trainable = trainable
        self.vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=self.max_tokens,
            output_mode=self.output_mode,
            output_sequence_length=self.output_sequence_length,
            vocabulary=self.vocabulary,
            standardize=self.standardize,
            trainable=self.trainable,
            # name="vectorizer",
        )

    def call(self, inputs):
        return self.vectorize_layer(inputs)
    
    def adapt(self, data):
        self.vectorize_layer.adapt(data)

    def get_vocabulary(self):
        return self.vectorize_layer.get_vocabulary()

    def get_config(self):
        config = super(TextVectorizer, self).get_config()
        config.update({
            "max_tokens": self.max_tokens,
            "output_mode": self.output_mode,
            "output_sequence_length": self.output_sequence_length,
            "standardize": self.standardize,
            "vocabulary": self.vocabulary,
        })
        return config

def filter_split_dataset(X_train, X_test, y_train, y_test, cols, validation_split=0.2):
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

    # Split the data into train and test
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=validation_split, stratify=y_train, random_state=SEED
    )
    data = (X_train_split, X_val_split, y_train_split, y_val_split, X_test, y_test)
    # Return the aligned data with features squeezed to remove the extra dimension if necessary
    return data

def create_mlflow_dataset(data):
    """
    Unpacks the provided data and creates MLflow datasets for training, validation, and testing.

    Args:
        data (tuple): A tuple containing the following elements:
            - X_train_split (pd.DataFrame): Training features.
            - X_val_split (pd.DataFrame): Validation features.
            - y_train_split (pd.Series): Training targets.
            - y_val_split (pd.Series): Validation targets.
            - X_test (pd.DataFrame): Testing features.
            - y_test (pd.Series): Testing targets.

    Returns:
        tuple: A tuple containing the following MLflow datasets:
            - train_mlflow: MLflow dataset for training.
            - val_mlflow: MLflow dataset for validation.
            - test_mlflow: MLflow dataset for testing.
    """
    # Unpack the data and create the datasets
    X_train_split, X_val_split, y_train_split, y_val_split, X_test, y_test = data
    train = X_train_split.assign(target=y_train_split)
    val = X_val_split.assign(target=y_val_split)
    test = X_test.assign(target=y_test)

    # Create mlflow datasets
    train_mlflow = mlflow.data.from_pandas(train, name="dataset", targets="target")
    val_mlflow = mlflow.data.from_pandas(val, name="dataset", targets="target")
    test_mlflow = mlflow.data.from_pandas(test, name="dataset", targets="target")

    # Return the mlflow datasets
    return train_mlflow, val_mlflow, test_mlflow

def to_tensorflow_dataset(
    data,
    batch_size=32
    ):
    """
    Converts training and testing data into TensorFlow datasets.

    Parameters:
    X_train (pd.DataFrame): Training features.
    X_test (pd.DataFrame): Testing features.
    y_train (pd.DataFrame): Training labels.
    y_test (pd.DataFrame): Testing labels.
    col_name (str): Column name to align splits with the corpus.
    path (str): Path to the parquet file.
    validation_split (float, optional): Fraction of the training data to be used as validation data. Default is 0.2.
    batch_size (int, optional): Number of samples per batch. Default is 32.

    Returns:
    tuple: A tuple containing three TensorFlow datasets (train_ds, val_ds, test_ds).
    """
    # Unzip the data into the training, validating and testing sets
    X_train_split, X_val_split, y_train_split, y_val_split, X_test, y_test = data

    # Create the tensorflow datasets for train, val and test
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_split, y_train_split)).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_split, y_val_split)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    # Cache the datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Return the tensorflow datasets
    return train_ds, val_ds, test_ds

def log_dataset_info(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Logs information about the provided datasets by saving them as CSV files and recording them as artifacts in MLflow.

    Parameters:
    X_train (array-like): Training feature dataset.
    y_train (array-like): Training target dataset.
    X_val (array-like): Validation feature dataset.
    y_val (array-like): Validation target dataset.
    X_test (array-like): Test feature dataset.
    y_test (array-like): Test target dataset.

    The function performs the following steps:
    1. Creates a directory named 'datasets' if it does not already exist.
    2. Saves each dataset (X_train, y_train, X_val, y_val, X_test, y_test) as a CSV file in the 'datasets' directory.
    3. Logs each CSV file as an artifact in MLflow.
    4. Logs the sizes of the training, validation, and test datasets as parameters in MLflow.
    """
    # Créer un répertoire temporaire pour enregistrer les datasets
    os.makedirs("datasets", exist_ok=True)

    # Enregistrer les datasets sous forme de fichiers CSV
    pd.DataFrame(X_train).to_csv("datasets/X_train.csv", index=False)
    pd.DataFrame(y_train).to_csv("datasets/y_train.csv", index=False)
    pd.DataFrame(X_val).to_csv("datasets/X_val.csv", index=False)
    pd.DataFrame(y_val).to_csv("datasets/y_val.csv", index=False)
    pd.DataFrame(X_test).to_csv("datasets/X_test.csv", index=False)
    pd.DataFrame(y_test).to_csv("datasets/y_test.csv", index=False)

    # Enregistrer les fichiers CSV en tant qu'artefacts dans MLflow
    mlflow.log_artifact("datasets/X_train.csv")
    mlflow.log_artifact("datasets/y_train.csv")
    mlflow.log_artifact("datasets/X_val.csv")
    mlflow.log_artifact("datasets/y_val.csv")
    mlflow.log_artifact("datasets/X_test.csv")
    mlflow.log_artifact("datasets/y_test.csv")

    # Enregistrer des informations supplémentaires sur les datasets
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("val_size", len(X_val))
    mlflow.log_param("test_size", len(X_test))

# get the loss by epoch
def plot_loss(history):
    """
    Plots the training and validation loss by epoch.

    Parameters:
    history (History): A History object returned by the fit method of a Keras model. 
                       It contains the training and validation loss values for each epoch.

    Returns:
    matplotlib.axes._subplots.AxesSubplot: The plot of the training and validation loss by epoch.
    """
    # Plot the loss by epoch
    history_df = pd.DataFrame(history.history)
    plot = history_df.loc[:, ["loss", "val_loss"]].plot()
    plot.set_xlabel("Epoch")
    plot.set_ylabel("Loss")
    plot.set_title("Loss by Epoch")
    return plot

# Fonctions - Embeddings matrix creation, model building and training
def create_embedding_matrix(vocab, pretrained_weights, initializer_method, initializer_range):
    """
    Creates an embedding matrix using pretrained weights and random initialization for out-of-vocabulary words.

    Args:
        vocab (list): List of words in the vocabulary.
        pretrained_weights (dict): Dictionary containing pretrained word embeddings.
        random_weights (str): Method to initialize random weights for out-of-vocabulary words. 
                              Accepts 'normal' or 'uniform'. Defaults to 'normal' if an invalid method is provided.

    Returns:
        tuple: A tuple containing:
            - embedding_matrix (np.ndarray): The embedding matrix where each row corresponds to a word in the vocabulary.
            - embedding_dim (int): The dimension of the embeddings.
    """
    embedding_dim = len(pretrained_weights['hello'])
    word_index = dict(zip(vocab, range(len(vocab))))

    # Init the embedding matrix
    match initializer_method:
            case 'normal':
                embedding_matrix = np.random.normal(loc=0.0, scale=initializer_range, size=(len(vocab), embedding_dim))
            case 'uniform':
                embedding_matrix = np.random.uniform(-initializer_range, initializer_range, size=(len(vocab), embedding_dim))
            case _:
                embedding_matrix = np.random.normal(loc=0.0, scale=0.1, size=(len(vocab), embedding_dim))
                print("Uniquement les valeurs normal ou uniform sont acceptées. Les poids ont été initialisés avec la méthode normale.")
    
    # Remplir la matrice d'embeddings avec les mots trouvés dans la vectorize layer et laisse les autres poids 
    for word, i in word_index.items():
        if i < len(word_index):
            try:
                embedding_vector = pretrained_weights[word]
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            except KeyError:
                pass
    return embedding_matrix, embedding_dim


def create_tf_model(train_ds, model_params, optimizer_params, weights_params):
    """
    Creates a TensorFlow model for text classification.
    Args:
        max_tokens (int): The maximum size of the vocabulary.
        seq_length (int): The length of the input sequences.
        embedding_dim (int): The dimension of the embedding vectors.
        additionnal_layers (list): A list of additional Keras layers to add to the model.
        pretrained_weights (str, optional): Path to the pretrained embedding weights. Defaults to None.
        random_weights (str, optional): Initializer for embedding weights if pretrained weights are not provided. Defaults to 'uniform'.
    Returns:
        tf.keras.Model: A compiled TensorFlow model ready for training.
    """
    # Unpack the model parameters
    max_tokens = model_params["max_tokens"]
    seq_length = model_params["seq_length"]
    embedding_dim = model_params["embedding_dim"]
    additionnal_layers = model_params["additionnal_layers"]
    embeddings_initializer = model_params["embeddings_initializer"]
    text_standardize_func = model_params["text_standardize_func"]

    # Unpack the weights parameters
    pretrained_weights = weights_params["pretrained_weights"]
    initializer_method = weights_params["initializer_method"]
    initializer_range = weights_params["initializer_range"]

    # Create the text vectorization layer
    vectorize_layer = TextVectorizer(
        max_tokens=max_tokens, 
        output_mode='int', 
        output_sequence_length=seq_length,
        standardize_func=text_standardize_func,
        trainable=False,
        vocabulary=None
    )

    # Adapt the text vectorization layer to the train dataset
    vectorize_layer.adapt(train_ds.map(lambda text, label: text))
    # Get the vocabulary
    vocab = vectorize_layer.get_vocabulary()
    vocab_size = len(vocab)
    print("Vocabulary size: ", vocab_size)
    
    # Create the embedding layer
    if pretrained_weights is not None:
        embedding_matrix, embedding_dim = create_embedding_matrix(vocab, pretrained_weights, initializer_method, initializer_range)
    # Create the embedding layer
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorize_layer,
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix] if pretrained_weights is not None else None,
            input_length=seq_length,
            embeddings_initializer=embeddings_initializer if pretrained_weights is None else None,
            trainable=True,
            name="embedding",
            ),
        *additionnal_layers,
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    # Unpack the optimizer parameters
    learning_rate = optimizer_params["learning_rate"]
    epsilon = optimizer_params["epsilon"]

    # Compile the model
    model.compile(
            loss=tf.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon),
            metrics=[tf.metrics.BinaryAccuracy(threshold=0.5), tf.metrics.AUC()],
        )
    
    return model