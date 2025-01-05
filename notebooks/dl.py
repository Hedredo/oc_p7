import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import numpy as np
from sklearn.model_selection import train_test_split

# Constantes
SEED = 314

# Clear custom objects in case of re-import
tf.keras.utils.get_custom_objects().clear()

@tf.keras.utils.register_keras_serializable(package="custom_text_func", name="custom_standardization")
def custom_standardization(tensor):
    tensor = tf.strings.lower(tensor)  # lowercase
    tensor = tf.strings.regex_replace(tensor, r"@\w+", " ")  # strip mentions
    tensor = tf.strings.regex_replace(tensor, r"http\S+|www\S+", " ")  # strip urls
    tensor = tf.strings.regex_replace(tensor, r"#\w+", " ")  # strip hashtags
    tensor = tf.strings.regex_replace(tensor, r"[^\w\s\d]", " ")  # strip punctuation
    tensor = tf.strings.regex_replace(tensor, r"\s{2,}", " ")  # strip multiple spaces
    return tf.strings.strip(tensor)  # strip leading and trailing spaces

@tf.keras.utils.register_keras_serializable(package="custom_layer", name="TextVectorizer")
class TextVectorizer(tf.keras.layers.Layer):
    def __init__(self, max_tokens=None, output_mode='int', output_sequence_length=None, standardize=None, trainable=False, vocabulary=None):
        super().__init__()
        self.max_tokens = max_tokens
        self.output_mode = output_mode
        self.output_sequence_length = output_sequence_length
        self.standardize = standardize
        self.vocabulary = vocabulary
        self.trainable = trainable
        self.vectorize_layer = TextVectorization(
            max_tokens=self.max_tokens,
            output_mode=self.output_mode,
            output_sequence_length=self.output_sequence_length,
            vocabulary=self.vocabulary,
            standardize=self.standardize,
            trainable=self.trainable,
            name="vectorizer",
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


def to_tensorflow_dataset(
    col_name,
    path,
    X_train, X_test, y_train, y_test,
    validation_split=0.2, batch_size=32
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
    # Align the splits with the corpus directly from dataframe
    X_train, X_test = filter_dataset(
        X_train,
        X_test,
        [col_name],
    )
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=validation_split, stratify=y_train, random_state=SEED
    )
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

# Fonctions - Embeddings matrix creation, model building and training
def create_embedding_matrix(vocab, pretrained_weights, random_weights):
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
    # Initialiser la matrice d'embeddings
    match random_weights:
        case 'normal':
            embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab), embedding_dim))
        case 'uniform':
            embedding_matrix = np.random.uniform(0, 1, size=(len(vocab), embedding_dim))
        case _:
            embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab), embedding_dim))
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


def create_tf_model(max_tokens, seq_length, text_standardize_func, embedding_dim, additionnal_layers, train_ds, pretrained_weights=None, random_weights='uniform'):
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
    # Create the text vectorization layer
    vectorize_layer = TextVectorizer(
        max_tokens=max_tokens,
        output_sequence_length=seq_length,
        standardize=text_standardize_func,
    )

    # Adapt the text vectorization layer to the train dataset
    vectorize_layer.adapt(train_ds.map(lambda text, label: text))
    # Get the vocabulary
    vocab = vectorize_layer.get_vocabulary()
    vocab_size = len(vocab)
    print("Vocabulary size: ", vocab_size)
    # Create the embedding layer
    if pretrained_weights is not None:
        embedding_matrix, embedding_dim = create_embedding_matrix(vocab, pretrained_weights, random_weights)
    # Create the embedding layer
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorize_layer,
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix] if pretrained_weights is not None else None,
            input_length=seq_length,
            embeddings_initializer=random_weights,
            trainable=True,
            name="embedding",
            ),
        *additionnal_layers,
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    # Compile the model
    model.compile(
            loss=tf.losses.BinaryCrossentropy(),
            optimizer="adam",
            metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)],
        )
    
    return model