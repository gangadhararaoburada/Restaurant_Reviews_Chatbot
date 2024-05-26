''' The specific topic or concept related to the provided code is sentiment analysis in Python using TextBlob, particularly in the context of analyzing restaurant reviews.'''
import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd

class Restaurant:
    def __init__(self):
        self.greetings = ["hello", "hi", "hey", "greetings", "sup", "what's up"]
        self.name = "Restaurant Owner"
        self.sentiment_counts = {"1": 0, "0": 0}

    def get_sentiment(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity <= 0:
            sentiment = "0"
        else:
            sentiment = "1"
        return sentiment

    def feedback(self):
        print(f"{self.name}: Hi! I am Restaurant Owner. How can I help you?")
        while True:
            try:
                user_input = next(self.input_iterator)
            except StopIteration:
                print(f"{self.name}: Goodbye!")
                break

            print("Customer:", user_input)

            if user_input in self.greetings:
                print(f"{self.name}: Hello! How can I assist you?")
                continue

            sentiment = self.get_sentiment(user_input)
            if sentiment == "1":
                print(f"Prediction: {sentiment} (You expressed a Positive Review)")

            else:
                print(f"Prediction: {sentiment} (You expressed a Negative Review)")

            self.sentiment_counts[sentiment] += 1

    def plot_sentiment_percentages(self):
       labels = ['Positive', 'Negative']  # Updated labels
       sizes = [self.sentiment_counts['1'], self.sentiment_counts['0']]
       colors = ['green', 'red']
       explode = (0.1, 0)

       plt.pie(sizes, explode = explode,labels=labels, colors=colors, autopct='%1.1f%%',shadow=True, startangle=140)
       plt.axis('equal')
       plt.title("Sentimental Analysis of Restaurant Reviews")
       plt.show()


if __name__ == "__main__":
    restaurant = Restaurant()
    data = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')
    restaurant.input_iterator = iter(data['Review'])
    restaurant.feedback()
    restaurant.plot_sentiment_percentages()