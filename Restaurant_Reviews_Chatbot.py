''' The specific topic or concept related to the provided code is sentiment analysis in Python using TextBlob, particularly in the context of analyzing restaurant reviews.'''

import nltk
import sys
import os
import re
import argparse
import logging
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Get the directory where the current .py file is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up logging to store log file in the same directory as the .py file
LOG_FILE = os.path.join(SCRIPT_DIR, 'Restaurant_Reviews.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def log_and_print(message, level='info'):
    print(message)
    if level == 'info':
        logging.info(message)
    elif level == 'error':
        logging.error(message)
    elif level == 'warning':
        logging.warning(message)

# Print Python version and compatibility status
log_and_print(f"Running on Python [ {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ]")
if sys.version_info >= (3, 12):
    print("     - This version is compatible and supported for the execution of the code.")
else:
    log_and_print("Error: ")
    log_and_print(f"    - The version [ {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ] is not compatible and not supported for the execution of the code.")
    log_and_print("    - This code requires Python [ 3.12 ] or later versions.")
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
            log_and_print(f"Data file not found: {self.data_file}")
            sys.exit(1)
        except Exception as e:
            log_and_print(f"Error loading data: {str(e)}")
            sys.exit(1)

    def feedback(self):
        """Process reviews and analyze sentiments."""
        log_and_print(f"{self.name}: Hi! I am Restaurant Owner. How can I help you?")
        
        data = self.load_data()
        self.input_iterator = iter(data['Review'])
        
        for review in tqdm(self.input_iterator, total=len(data), desc="Processing reviews"):
            try:
                log_and_print("Customer:", review)
                
                if review.lower() in self.greetings:
                    log_and_print(f"{self.name}: Hello! How can I assist you?")
                    continue

                sentiment, polarity = self.get_sentiment(review)
                self.sentiment_counts[sentiment] += 1
                self.processed_reviews.append({
                    'review': review,
                    'sentiment': sentiment,
                    'polarity': polarity
                })
                
                log_and_print(f"Prediction: {sentiment} (Polarity: {polarity:.3f})")
                logging.info(f"Review: {review[:50]}... | Sentiment: {sentiment} | Polarity: {polarity:.3f}")
                
            except Exception as e:
                log_and_print(f"Error processing review: {str(e)}")
                continue

        log_and_print(f"{self.name}: Goodbye!")
        self.save_results()

    def save_results(self):
        """Save analysis results to CSV."""
        try:
            results_df = pd.DataFrame(self.processed_reviews)
            LOG_FILE = os.path.join(SCRIPT_DIR, f'sentiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            output_file = LOG_FILE
            results_df.to_csv(output_file, index=False)
            log_and_print(f"Results saved to {output_file}")
        except Exception as e:
            log_and_print(f"Error saving results: {str(e)}")

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
            plt.savefig(os.path.join(SCRIPT_DIR, 'sentiment_pie_chart.png'))
            plt.close()
            logging.info("Sentiment pie chart saved as sentiment_pie_chart.png")
        except Exception as e:
            log_and_print(f"Error plotting sentiment chart: {str(e)}")

    def print_summary(self):
        """Print statistical summary of sentiment analysis."""
        try:
            total_reviews = sum(self.sentiment_counts.values())
            log_and_print("\nSentiment Analysis Summary:")
            log_and_print(f"Total Reviews: {total_reviews}")
            for sentiment, count in self.sentiment_counts.items():
                percentage = (count / total_reviews * 100) if total_reviews > 0 else 0
                log_and_print(f"{sentiment}: {count} reviews ({percentage:.1f}%)")
            
            if self.sentiment_scores:
                log_and_print(f"\nPolarity Statistics:")
                log_and_print(f"Average Polarity: {np.mean(self.sentiment_scores):.3f}")
                log_and_print(f"Min Polarity: {np.min(self.sentiment_scores):.3f}")
                log_and_print(f"Max Polarity: {np.max(self.sentiment_scores):.3f}")
                log_and_print(f"Std Dev: {np.std(self.sentiment_scores):.3f}")
            
            logging.info("Summary statistics generated")
        except Exception as e:
            log_and_print(f"Error generating summary: {str(e)}")

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
        log_and_print(f"Fatal error in main execution: {str(e)}")
        sys.exit(1)
