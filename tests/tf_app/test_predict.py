import unittest
import tensorflow as tf
from tf_app.model import custom_standardization, load_model

class TestPredict(unittest.TestCase):

    def setUp(self):
        self.model = load_model()

    def test_custom_standardization(self):
        input_text = tf.constant(["Hello @user! Check out http://example.com"])
        expected_output = tf.constant(["hello check out"])
        output = custom_standardization(input_text)
        self.assertTrue(tf.reduce_all(tf.equal(output, expected_output)))

    def test_model_prediction_positive(self):
        input_text = tf.constant(["I love this product!"])
        probability = self.model.predict(input_text)
        sentiment = "positive" if probability[0][0] > 0.5 else "negative"
        self.assertEqual(sentiment, "positive")

    def test_model_prediction_negative(self):
        input_text = tf.constant(["I hate this product!"])
        probability = self.model.predict(input_text)
        sentiment = "positive" if probability[0][0] > 0.5 else "negative"
        self.assertEqual(sentiment, "negative")
        

if __name__ == '__main__':
    unittest.main()