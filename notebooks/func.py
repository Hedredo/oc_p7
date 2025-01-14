# Description: Custom functions for text preprocessing
import tensorflow as tf
import mlflow.spacy
from typing import List

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
