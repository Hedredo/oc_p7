import mlflow.tensorflow
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="custom_text_func", name="custom_standardization")
def custom_standardization(tensor):
    tensor = tf.strings.lower(tensor)  # lowercase
    tensor = tf.strings.regex_replace(tensor, r"@\w+", " ")  # strip mentions
    tensor = tf.strings.regex_replace(tensor, r"http\S+|www\S+", " ")  # strip urls
    tensor = tf.strings.regex_replace(tensor, r"[^\w\s\d]", " ")  # strip punctuation
    tensor = tf.strings.regex_replace(tensor, r"\s{2,}", " ")  # strip multiple spaces
    return tf.strings.strip(tensor)  # strip leading and trailing spaces

# Load the model
model_path = "./mlflow/689416981458083287/b5efda2cee954ff1a88923f357bc0525/artifacts/model"
model = mlflow.tensorflow.load_model(model_path) # Load the model with the MLflow Keras API

# Predict
prediction = model.predict(tf.constant([data.text]))
label = "positive" if prediction[0][0] > 0.5 else "negative"
print({"prediction": label, "probability": float(prediction[0][0])})