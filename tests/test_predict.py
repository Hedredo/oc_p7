import unittest
from container.model import predict_sentiment

class TestPredict(unittest.TestCase):

    def test_predict_sentiment_positive(self):
        text = "I am happy today!"
        label = predict_sentiment(text)
        self.assertEqual(label, "positive")
    
    def test_predict_sentiment_negative(self):
        text = "I am unhappy today!"
        label = predict_sentiment(text)
        self.assertEqual(label, "negative")
    
    def test_predict_sentiment_neutral(self):
        text = "I am neutral today!"
        label = predict_sentiment(text)
        self.assertEqual(label, "neutral")

if __name__ == '__main__':
    unittest.main()