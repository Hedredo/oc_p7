import unittest
import tensorflow as tf
from predict import custom_standardization, model

class TestPredict(unittest.TestCase):

    def test_custom_standardization(self):
        input_text = tf.constant(["Hello @user! Check out http://example.com"])
        expected_output = tf.constant(["hello check out"])
        output = custom_standardization(input_text)
        self.assertTrue(tf.reduce_all(tf.equal(output, expected_output)))

    def test_model_prediction_positive(self):
        input_text = tf.constant(["I love this product!"])
        prediction = model.predict(input_text)
        label = "positive" if prediction[0][0] > 0.5 else "negative"
        self.assertEqual(label, "positive")

    def test_model_prediction_negative(self):
        input_text = tf.constant(["I hate this product!"])
        prediction = model.predict(input_text)
        label = "positive" if prediction[0][0] > 0.5 else "negative"
        self.assertEqual(label, "negative")

if __name__ == '__main__':
    unittest.main()