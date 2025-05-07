# Restaurant Review Sentiment Analysis

This project performs sentiment analysis on restaurant reviews stored in a TSV file, classifying them as **Positive**, **Neutral**, or **Negative**. It utilizes the **TextBlob** library for sentiment analysis, provides visual and statistical summaries, and supports command-line usage for flexible integration.

---

## 🚀 Features

- **Sentiment Analysis**: Classifies reviews based on polarity:
  - Positive: 'polarity > 0.1'
  - Neutral: '-0.1 <= polarity <= 0.1'
  - Negative: 'polarity < -0.1'
- **Data Preprocessing**: Cleans review text (removes special characters, normalizes spacing).
- **Visualization**: Generates a pie chart ('sentiment_pie_chart.png') showing sentiment distribution.
- **Result Export**: Saves output with sentiments and polarity to a timestamped CSV file.
- **Progress Tracking**: Displays a progress bar using 'tqdm' while processing.
- **Logging**: Saves logs to a timestamped log file for debugging and traceability.
- **Statistical Summary**: Outputs sentiment counts, percentages, and polarity stats (mean, min, max, std dev).
- **Command-Line Interface**: Allows setting input file path using the '--data' argument.

---

## 🧰 Prerequisites

- **Python**: Version '3.12' or later
- **Input File**: A '.tsv' file (e.g., 'Restaurant_Reviews.tsv') with a 'Review' column containing text data.

---

## 🛠 Installation

### 1. Clone the Repository

'''bash
git clone <repository-url>
cd <repository-directory>
'''

### 2. Create a Virtual Environment (Recommended)

'''bash
python -m venv venv
# Activate the environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
'''

### 3. Install Dependencies

Create a 'requirements.txt' with the following content:

'''
nltk==3.8.1
textblob==0.17.1
matplotlib==3.8.2
pandas==2.2.2
tqdm==4.66.2
numpy==1.26.4
'''

Then install:

'''bash
pip install -r requirements.txt
'''

### 4. Download NLTK Data

Run the following Python commands to download required corpora:

'''python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
'''

---

## 📄 Usage

### Prepare the Input File

Ensure the input '.tsv' file has a column named 'Review'. Example format:

'''
Review	Liked
Wow... Loved this place.	1
Crust is not good.	0
'''

### Run the Script

'''bash
python script.py --data /path/to/Restaurant_Reviews.tsv
'''

> If '--data' is not specified, it defaults to:
> 'E:/Upgrades/Restaurant Reviews/Restaurant_Reviews.tsv'

### View Help

'''bash
python script.py --help
'''

---

## 📦 Outputs

### ✅ Console Output

- Shows each review's sentiment and polarity.
- Displays a progress bar during processing.
- Prints a sentiment summary and polarity statistics.

### 📊 Pie Chart

- A file 'sentiment_pie_chart.png' is saved showing sentiment distribution (Positive in green, Neutral in yellow, Negative in red).

### 📁 CSV Output

- A timestamped CSV file (e.g., 'sentiment_results_20250507_123456.csv') with:
  '''
  review,sentiment,polarity
  "Wow... Loved this place.",Positive,0.600
  "Crust is not good.",Negative,-0.350
  '''

### 📃 Log File

- A log file (e.g., 'sentiment_analysis_20250507_123456.log') with detailed logs and error tracking.

---

## 📈 Example Output

'''
Running on Python [ 3.12.0 ]
- This version is compatible and supported.
Restaurant Owner: Hi! I am Restaurant Owner. How can I help you?

Processing reviews: 100%|██████████| 1000/1000 [00:05<00:00, 180.12it/s]

Customer: Wow... Loved this place.
Prediction: Positive (Polarity: 0.600)
...

Restaurant Owner: Goodbye!
Results saved to sentiment_results_20250507_123456.csv
'''

### Summary Example:

'''
Sentiment Analysis Summary:
Total Reviews: 1000
Positive: 500 reviews (50.0%)
Neutral: 200 reviews (20.0%)
Negative: 300 reviews (30.0%)

Polarity Statistics:
Average Polarity: 0.150
Min Polarity: -0.800
Max Polarity: 0.900
Std Dev: 0.350
'''

---

## 🧩 Troubleshooting

- **FileNotFoundError**: Ensure the TSV file exists at the provided path or use '--data' to specify it.
- **NLTK Errors**: If NLTK data is missing, re-run the 'nltk.download()' commands in the installation section.
- **Python Version**: Confirm Python is version 3.12+: 'python --version'.
- **Dependency Issues**: Use 'pip list' to verify installed packages and versions.

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to open issues or submit pull requests to improve this project.

---

## 📜 License
