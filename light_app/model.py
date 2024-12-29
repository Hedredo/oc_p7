import re

# Add

# Define the function to predict the sentiment of the input text
def predict_sentiment(text: str) -> str:
    """
    Predict the sentiment of the input text.

    Parameters:
    text (str): The input text.

    Returns:
    str: The sentiment label.
    """
    # Define the regular expression patterns for positive and negative words
    pos_pattern = r"good|happy|cool|best|better|nice|excellent|positive|fortunate|correct|superior|great|positive|superb|wonderful|awesome|fantastic|terrific|amazing|incredible|fabulous|marvelous|excellent|outstanding|exceptional|perfect|pleasing|delightful|pleasurable|satisfying|acceptable|agreeable|enjoyable|favorable|good|gratifying|great|pleasing|positive"
    neg_pattern = r"bad|unhappy|horrible|worst|negative|unfortunate|wrong|inferior|poor|negative|dreadful|terrible|awful|atrocious|abysmal|appalling|dreadful|lousy|unsatisfactory|unacceptable|disagreeable|displeasing|unfavorable|unpleasant|bad|disgusting|distasteful|foul|gross|nasty|nauseating|obnoxious|offensive|repellent|repulsive|revolting|vile|wretched|bad|disagree"

    # Compile the regular expression pattern into a regular expression object
    pos_re = re.compile(pos_pattern, re.IGNORECASE)
    neg_re = re.compile(neg_pattern, re.IGNORECASE)
    
    # Count the number of positive and negative words in the input
    pos_count = len(pos_re.findall(text))
    neg_count = len(re.findall(neg_re, text))
    # Compute the sentiment score
    sentiment_score = pos_count - neg_count
    # Match the sentiment score to a sentiment category
    if sentiment_score > 0:
        label = "positive"
    elif sentiment_score < 0:
        label = "negative"
    else:
        label = "neutral"
    return label