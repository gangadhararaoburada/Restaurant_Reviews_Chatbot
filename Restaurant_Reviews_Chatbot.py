''' The specific topic or concept related to the provided code is sentiment analysis in Python using TextBlob, particularly in the context of analyzing restaurant reviews.'''

import nltk
import sys
import re
import argparse
import logging
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename=f'sentiment_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Print Python version and compatibility status
print(f"Running on Python [ {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ]")
if sys.version_info >= (3, 12):
    print("     - This version is compatible and supported for the execution of the code.")
else:
    print("Error: ")
    print(f"    - The version [ {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ] is not compatible and not supported for the execution of the code.")
    print("    - This code requires Python [ 3.12 ] or later versions.")
    sys.exit(1)

class Restaurant:
    def __init__(self, data_file):
        self.greetings = ["hello", "hi", "hey", "greetings", "sup", "what's up"]
        self.name = "Restaurant Owner"
        self.sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        self.data_file = data_file
        self.sentiment_scores = []
        self.processed_reviews = []

    def clean_text(self, text):
        """Clean text by removing special characters and extra spaces."""
        try:
            text = re.sub(r'[^\w\s]', '', text.lower())
            text = re.sub(r'\s+', ' ', text.strip())
            return text
        except Exception as e:
            logging.error(f"Error cleaning text: {str(e)}")
            return text

    def get_sentiment(self, text):
        """Analyze sentiment with more nuanced categories."""
        try:
            cleaned_text = self.clean_text(text)
            blob = TextBlob(cleaned_text)
            polarity = blob.sentiment.polarity
            self.sentiment_scores.append(polarity)

            if polarity > 0.1:
                sentiment = "Positive"
            elif polarity < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            return sentiment, polarity
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {str(e)}")
            return "Neutral", 0.0

    def load_data(self):
        """Load and validate data from TSV file."""
        try:
            data = pd.read_csv(self.data_file, delimiter='\t')
            if 'Review' not in data.columns:
                raise ValueError("Input file must contain 'Review' column")
            return data
        except FileNotFoundError:
            logging.error(f"Data file not found: {self.data_file}")
            print(f"Error: Data file {self.data_file} not found")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            print(f"Error loading data: {str(e)}")
            sys.exit(1)

    def feedback(self):
        """Process reviews and analyze sentiments."""
        print(f"{self.name}: Hi! I am Restaurant Owner. How can I help you?")
        
        data = self.load_data()
        self.input_iterator = iter(data['Review'])
        
        for review in tqdm(self.input_iterator, total=len(data), desc="Processing reviews"):
            try:
                print("Customer:", review)
                
                if review.lower() in self.greetings:
                    print(f"{self.name}: Hello! How can I assist you?")
                    continue

                sentiment, polarity = self.get_sentiment(review)
                self.sentiment_counts[sentiment] += 1
                self.processed_reviews.append({
                    'review': review,
                    'sentiment': sentiment,
                    'polarity': polarity
                })
                
                print(f"Prediction: {sentiment} (Polarity: {polarity:.3f})")
                logging.info(f"Review: {review[:50]}... | Sentiment: {sentiment} | Polarity: {polarity:.3f}")
                
            except Exception as e:
                logging.error(f"Error processing review: {str(e)}")
                print(f"Error processing review: {str(e)}")
                continue

        print(f"{self.name}: Goodbye!")
        self.save_results()

    def save_results(self):
        """Save analysis results to CSV."""
        try:
            results_df = pd.DataFrame(self.processed_reviews)
            output_file = f'sentiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            results_df.to_csv(output_file, index=False)
            logging.info(f"Results saved to {output_file}")
            print(f"Results saved to {output_file}")
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            print(f"Error saving results: {str(e)}")

    def plot_sentiment_percentages(self):
        """Plot sentiment distribution as a pie chart."""
        try:
            labels = list(self.sentiment_counts.keys())
            sizes = list(self.sentiment_counts.values())
            colors = ['green', 'yellow', 'red']
            explode = (0.1, 0, 0)

            plt.figure(figsize=(8, 6))
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
                    autopct='%1.1f%%', shadow=True, startangle=140)
            plt.axis('equal')
            plt.title("Sentiment Analysis of Restaurant Reviews")
            plt.savefig('sentiment_pie_chart.png')
            plt.close()
            logging.info("Sentiment pie chart saved as sentiment_pie_chart.png")
        except Exception as e:
            logging.error(f"Error plotting sentiment chart: {str(e)}")
            print(f"Error plotting sentiment chart: {str(e)}")

    def print_summary(self):
        """Print statistical summary of sentiment analysis."""
        try:
            total_reviews = sum(self.sentiment_counts.values())
            print("\nSentiment Analysis Summary:")
            print(f"Total Reviews: {total_reviews}")
            for sentiment, count in self.sentiment_counts.items():
                percentage = (count / total_reviews * 100) if total_reviews > 0 else 0
                print(f"{sentiment}: {count} reviews ({percentage:.1f}%)")
            
            if self.sentiment_scores:
                print(f"\nPolarity Statistics:")
                print(f"Average Polarity: {np.mean(self.sentiment_scores):.3f}")
                print(f"Min Polarity: {np.min(self.sentiment_scores):.3f}")
                print(f"Max Polarity: {np.max(self.sentiment_scores):.3f}")
                print(f"Std Dev: {np.std(self.sentiment_scores):.3f}")
            
            logging.info("Summary statistics generated")
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            print(f"Error generating summary: {str(e)}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Restaurant Review Sentiment Analysis')
    parser.add_argument('--data', default='E:/Upgrades/Restaurant Reviews/Restaurant_Reviews.tsv', 
                       help='Path to the TSV file containing restaurant reviews')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    try:
        restaurant = Restaurant(args.data)
        restaurant.feedback()
        restaurant.plot_sentiment_percentages()
        restaurant.print_summary()
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
