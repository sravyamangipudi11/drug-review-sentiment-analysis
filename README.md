<xaiArtifact artifact_id="74bfc0ae-9b90-4d3d-8c78-0ed4ce4fbbb4" artifact_version_id="e3afa821-0ee7-4f50-8665-563df985d5f6" title="README.md" contentType="text/markdown">

# Drug Review Sentiment Analysis

This project analyzes patient drug reviews to predict sentiment (positive or negative) using a LightGBM classifier, achieving an accuracy of 89.01%. The dataset, sourced from `drugsComTrain_raw.csv` and `drugsComTest_raw.csv`, contains 215,063 reviews with features like drug names, conditions, ratings, and review text. The project includes extensive exploratory data analysis (EDA), feature engineering, and natural language processing (NLP) to uncover patterns and build a robust sentiment prediction model.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [EDA Visualizations](#eda-visualizations)
- [Model Performance](#model-performance)
- [Feature Engineering](#feature-engineering)
- [Preprocessing](#preprocessing)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal is to classify patient drug reviews as positive (rating ≥ 5) or negative (rating < 5) using a LightGBM classifier. The project includes:
- **Exploratory Data Analysis (EDA):** Visualizing drug popularity, rating distributions, and condition prevalence.
- **Feature Engineering:** Creating 20+ features including sentiment scores, word counts, and encoded categorical variables.
- **NLP Preprocessing:** Cleaning review text with lowercasing, ASCII filtering, stopword removal, and Snowball stemming.
- **Model Training:** Using LightGBM with tuned hyperparameters for high accuracy and robust performance.

## Dataset
The dataset combines `drugsComTrain_raw.csv` and `drugsComTest_raw.csv`, totaling 215,063 reviews. Key columns include:
- `drugName`: Name of the drug.
- `condition`: Medical condition treated.
- `review`: Patient review text.
- `rating`: Rating (1–10).
- `date`: Review date.
- `usefulCount`: Number of users who found the review useful.

## Features
- **Model Accuracy:** 89.01% on 64,161 test samples.
- **Classification Metrics:**
  - Precision: 0.90 (positive), 0.84 (negative).
  - Recall: 0.96 (positive), 0.69 (negative).
  - F1-Score: 0.93 (positive), 0.76 (negative).
- **Key Visualizations:**
  - Word clouds for drug names and reviews.
  - Bar plots for top-rated and low-rated drugs.
  - Donut chart for rating distribution.
  - Countplots and histograms for temporal and rating analysis.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/drug-review-sentiment-analysis.git
   cd drug-review-sentiment-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Python 3.8+ and required libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `wordcloud`, `textblob`, `nltk`, `lightgbm`, `scikit-learn`, `joblib`).

## Usage
1. Place `drugsComTrain_raw.csv` and `drugsComTest_raw.csv` in the `data/` directory.
2. Run the Jupyter notebook:
   ```bash
   jupyter notebook drug_review_analysis.ipynb
   ```
3. The notebook includes:
   - Data loading and combination.
   - EDA with visualizations.
   - Feature engineering and preprocessing.
   - Model training and evaluation.
   - Saving the trained model and encoders (`lgbm_model.pkl`, `label_encoder_drug.pkl`, `label_encoder_condition.pkl`).

## EDA Visualizations
- **Word Clouds:** Visualize frequent drug names and review terms (positive/negative sentiments).
- **Bar Plots:** Show top 20 drugs with 10/10 and 1/10 ratings, and top 10 conditions.
- **Donut Chart:** Displays rating distribution (e.g., 31.6% for rating 10).
- **Countplots & Histograms:** Analyze reviews by year, month, day, and rating.
- **Correlation Heatmap:** Highlights relationships between engineered features.

## Model Performance
- **LightGBM Classifier:**
  - Hyperparameters: `n_estimators=10000`, `learning_rate=0.10`, `max_depth=7`, `subsample=0.9`.
  - Accuracy: 89.01%.
  - Confusion Matrix:
    ```
    [[11159  4915]
     [ 2134 45953]]
    ```
  - Classification Report:
    ```
                   precision    recall  f1-score   support
         0.0       0.84      0.69      0.76     16074
         1.0       0.90      0.96      0.93     48087
    accuracy                           0.89     64161
    ```

## Feature Engineering
- **Text Features:**
  - Word count, unique word count, letter count, punctuation count, uppercase/title case counts.
  - Mean word length, stopword count.
- **Sentiment Features:**
  - TextBlob polarity for raw and cleaned reviews.
  - Binary sentiment (1 for rating ≥ 5, 0 for rating < 5).
- **Categorical Encoding:**
  - Label encoding for `drugName` and `condition`.
  - Frequency encoding for `drugName` and `condition`.
  - Mean sentiment encoding for `drugName` and `condition`.
- **Temporal Features:**
  - Extracted year, month, and day from review dates.

## Preprocessing
- **Text Cleaning:**
  - Lowercasing, removing special characters, non-ASCII characters, and multiple spaces.
  - Replacing `&#039;` and multiple dots.
- **NLP Pipeline:**
  - Stopword removal using NLTK.
  - Snowball stemming for root word forms.
  - Sentiment analysis with TextBlob.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

</xaiArtifact>
