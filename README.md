## Fake-News-Detection-Via-Text-Classification
Fake News Detection - https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=fake.csv

### Dataset Access:
You can access the fake news + real news dataset from the provided link.


### Task 1 – Exploring Essential Information from Text Data and Preprocessing:
- Exploreed essential information from the textual data, such as the most commonly used words in the collection, real news, and fake news.
- Implemented "stopword removal" and "lemmatization" functions before calculating word frequency.
- Answered the following questions:
    1. What are the most commonly used words (top 100) in the collection, real news, and fake news?
    2. Can you easily distinguish between real news and fake news by reading the preprocessed textual data? What does the strongest feature set for machine learning look like?

### Task 2 – Building Machine Learning Model:
- Built machine learning text classification models to categorize news into "real" and "fake" categories.
- Split the data into training and testing collections (e.g., 70% and 30%).
- Experimented with different algorithms (e.g., regression and MultinomialNB) and feature sets (e.g., "term-frequency" and "TFIDF").
- Reported the performance of different algorithms in the provided table.
- Provided error analysis for the best-performing results (top 2) using "Test Confusion Matrix" and explain the outcome.

### Task 3 – Enhanced NLP Features:
- Implemented "POS Tagging" to locate specific kinds of words in the collection (e.g., nouns and verbs).
- Built additional classifier(s) leveraging POS information, such as using only "nouns" or "adj" + "noun" as features.
- Reported the performance of different algorithms in the provided table.
- Analyzed whether there is a performance improvement compared to Task 2 results and explain the outcome.

### Task 4 – Future Work:
- Proposed ideas for further enhancing the performance of machine learning models, such as exploring novel features or enhancing learning models.
- Designed a simple experiment using GPT (or GPT API) to investigate the potential use of GPT in addressing this problem.
