import pandas as pd
import spacy
from textblob import TextBlob

# Load language model.
nlp = spacy.load('en_core_web_md')

# Load the dataset.
df = pd.read_csv('amazon_product_reviews.csv')
# Preprocess the text data.
df['reviews.text'] = df['reviews.text'].astype(str).str.lower().str.strip()

# Create a function for sentiment analysis.
def analyse_sentiment_textblob(review):
    polarity_score = TextBlob(review).sentiment.polarity
    
    if polarity_score > 0:
        sentiment = 'positive'
    
    elif polarity_score < 0:
        sentiment = 'negative'
    
    else:
        sentiment = 'neutral'
    
    return sentiment

# Take two random reviews as samples.
sample_reviews = df['reviews.text'].sample(n=2, random_state=1)  
print("\nSentiment Analysis using TextBlob:")

# Display the randomly selected reviews along with their sentiments.
for review in sample_reviews:
    sentiment = analyse_sentiment_textblob(review)
    print(f"Review: '{review}' This review has as a {sentiment} sentiment.")
    print()

processed_reviews = [nlp(review) for review in sample_reviews]

# Calculate and print the similarity score
similarity_score = processed_reviews[0].similarity(processed_reviews[1])
print(f"\nSimilarity score between the two reviews: {similarity_score}")


