import unittest
import tensorflow as tf
from tf_app.model import custom_standardization_punct, set_model, SpacyTokenizer

class TestPredict(unittest.TestCase):

    def setUp(self):
        self.model = set_model("./models/neuralnet")
        self.tokenizer = SpacyTokenizer("./models/spacy_en_core_web_sm")

    def test_custom_standardization_punct(self):
        input_text = tf.constant(["Hello check out http://example.com"])
        expected_output = tf.constant(["hello check out"])
        output = custom_standardization_punct(input_text)
        self.assertTrue(tf.reduce_all(tf.equal(output, expected_output)))

    def test_model_prediction_positive(self):
        text = "I love this product!"
        probability = self.model.predict(self.tokenizer.tokenize([text]))
        sentiment = "positive" if probability[0][0] > 0.5 else "negative"
        self.assertEqual(sentiment, "positive")

    def test_model_prediction_negative(self):
        text = "I hate this product!"
        probability = self.model.predict(self.tokenizer.tokenize([text]))
        sentiment = "positive" if probability[0][0] > 0.5 else "negative"
        self.assertEqual(sentiment, "negative")
        

if __name__ == '__main__':
    unittest.main()