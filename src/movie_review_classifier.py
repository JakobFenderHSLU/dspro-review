import json
import re
from statistics import mean

import joblib
import nltk
import numpy as np
import streamlit
from keras.models import load_model
from nltk.sentiment import SentimentIntensityAnalyzer
import os

nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')


HTML_TAG_REGEX = r'<[^>]+>'
NON_ALPHABETIC_CHARS = r'[^a-zA-Z\s]'
STOPWORDS = nltk.corpus.stopwords.words("english")

NEGATIVE_WEIGHT = 1.6
POSITIVE_WORD_WEIGHT = 0.199
NEGATIVE_WORD_WEIGHT = 0.09
VECTOR_SIZE = 10000
sia = SentimentIntensityAnalyzer()

from pathlib import Path

RELATIVE_RESOURCE_DIR = Path("./resources/movie-review-classifier")
print(os.listdir(RELATIVE_RESOURCE_DIR))

classifiers = [
    "BernoulliNB",
    "ComplementNB",
    "MultinomialNB",
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "LogisticRegression",
    "MLPClassifier",
    "AdaBoostClassifier"
]


def read_list_from_file(filename):
    with open(filename) as file:
        return [line.strip() for line in file]


TOP_POS_WORDS = read_list_from_file(RELATIVE_RESOURCE_DIR / "top_pos_words.txt")
TOP_NEG_WORDS = read_list_from_file(RELATIVE_RESOURCE_DIR / "top_pos_words.txt")


confidence_dict = {}
with open(RELATIVE_RESOURCE_DIR / "confidence-dictionary.json", 'r') as file:
    confidence_dict = json.load(file)

word_ranking = {}
with open(RELATIVE_RESOURCE_DIR / "word-ranking.json", 'r') as file:
    word_ranking = json.load(file)


def prepare_text(text):
    text = re.sub(HTML_TAG_REGEX, '', text)
    positive_scores = list()
    negative_scores = list()
    top_pos_count = 0
    top_neg_count = 0

    for sentence in nltk.sent_tokenize(text):
        polarity_scores = sia.polarity_scores(sentence)
        positive_scores.append(polarity_scores["pos"])
        negative_scores.append(polarity_scores["neg"])
        words = nltk.word_tokenize(sentence)
        for word in words:
            if word.lower() in TOP_POS_WORDS:
                top_pos_count += 1
            if word.lower() in TOP_NEG_WORDS:
                top_neg_count += 1

    # Adding 1 to the final compound score to always have positive numbers
    # since some classifiers you'll use later don't work with negative numbers.
    avg_positive_scores = mean(positive_scores)
    avg_negative_scores = mean(negative_scores)
    contrast_score = avg_positive_scores - avg_negative_scores

    return {
        "avg_positive_scores": avg_positive_scores,
        "avg_negative_scores": avg_negative_scores,
        "top_pos_count": top_pos_count,
        "top_neg_count": top_neg_count,
        "contrast_score": contrast_score
    }


def get_sentiment_for_review_custom(texts):
    prepare_texts = [prepare_text(text) for text in texts]
    results = []
    for data in prepare_texts:
        contrast_score = round(data["contrast_score"], 3)
        positive_calculation = data["avg_positive_scores"] + (data["top_pos_count"] * POSITIVE_WORD_WEIGHT)
        negative_calculation = (data["avg_negative_scores"] * NEGATIVE_WEIGHT) + (
                data["top_neg_count"] * NEGATIVE_WORD_WEIGHT)

        if positive_calculation > negative_calculation:
            sentiment = "positive"
        else:
            sentiment = "negative"

        confidence = confidence_dict[str(contrast_score)]

        result_tuple = (sentiment, confidence)
        results.append(result_tuple)

    return [{
        "sentiment": result[0],
        "confidence": result[1]
    } for result in results]


def get_sentiment_for_review_ootb(texts, model):
    print(texts)
    prepared_texts = [prepare_text(text.lower()) for text in texts]
    format_data = [[data["avg_positive_scores"], data["avg_negative_scores"], data["contrast_score"] + 1,
                    data["top_pos_count"], data["top_neg_count"]]
                   for data in prepared_texts]
    results = model.predict(format_data)
    return [{
        "sentiment": result
    } for result in results]


def get_sentiment_for_review_nn(texts, model):
    cleaned_texts = [re.sub(HTML_TAG_REGEX, '', text) for text in texts]
    cleaned_texts = [re.sub(NON_ALPHABETIC_CHARS, ' ', text) for text in cleaned_texts]

    cleaned_texts_array = [nltk.word_tokenize(text.lower()) for text in cleaned_texts]

    # STOPWORD examples: "of", "a", "the"
    cleaned_texts_array = [[word for word in text_array if word not in STOPWORDS] for text_array in cleaned_texts_array]

    word_ranking_index = []

    for text_array in cleaned_texts_array:
        temp = []
        for word in text_array:
            if word in word_ranking and -1 < word_ranking[word] < 10000:
                temp.append(word_ranking[word])
        word_ranking_index.append(temp)

    vectorized_data = np.zeros((len(word_ranking_index), VECTOR_SIZE))
    for i, sequence in enumerate(word_ranking_index):
        vectorized_data[i, sequence] = 1

    results = model.predict(vectorized_data, batch_size=1, verbose=0)
    return_value = []

    for result in results:
        if result[0] > 0.5:
            sentiment = "positive"
        else:
            sentiment = "negative"
        return_value.append({
            "sentiment": sentiment,
            "confidence": abs((result[0] - 0.5) * 2)
        })
    return return_value


def movie_review_get_sentiment(texts: list, method: str):

    if texts[0] is None:
        raise ValueError("Missing text parameter")

    if type(texts[0]) is not str:
        raise ValueError("texts is not a string")

    if method == "custom":
        return get_sentiment_for_review_custom(texts)[0]

    elif method in classifiers:
        model = joblib.load(RELATIVE_RESOURCE_DIR / f"{method}.joblib")
        return get_sentiment_for_review_ootb(texts, model)[0]
    else:
        model = load_model(str(RELATIVE_RESOURCE_DIR / "sequential-model.keras"))
        return get_sentiment_for_review_nn(texts, model)[0]
