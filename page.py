import streamlit as st
from src import movie_review_classifier, product_review_classifier

import os

st.set_page_config(page_title="DSPRO DEMO", page_icon="🎬")

""" # Review Sentiment Classifier"""
""" This project was created by Jakob Fender as part of the course DSPRO-1 at the Lucerne University of Applied Sciences
 and Arts. The goal of this project is to create a classifier that can predict the sentiment of a review. This project
is split into two parts:
 
 
## Movie Review Classifier
The first part is a classifier for movie reviews. It uses a dataset of 50'000 movie reviews from IMDB. The classifier 
is trained on 40'000 reviews and tested on 10'000 reviews.It returns a simple positive or negative score for each 
review. The classifier can be trained with different methods. The following methods are available:
- Custom
- Out of the box
- Neural Network
"""

st.write("<br>", unsafe_allow_html=True)

cols = st.columns(2)

with cols[0]:
    method = st.selectbox("Select a method", ["custom", "Out of the box", "Neural Network"], 0)

with cols[1]:
    if method == "Out of the box":
        options = ["BernoulliNB", "ComplementNB", "MultinomialNB", "KNeighborsClassifier",
                   "LogisticRegression", "MLPClassifier"]
        method = st.selectbox("Select a method", options, 0)

text = st.text_area("Enter a review", "This movie was great!", key="movie")

if st.button("Submit", key="movie_submit"):
    # catch exception
    try:
        response = movie_review_classifier.movie_review_get_sentiment([text], method)
        st.write(response["sentiment"])
    except Exception as e:

        print(e)
        st.error("An error occurred " + str(e))

        # print stack trace
        import traceback
        traceback.print_exc()

st.write("<br>", unsafe_allow_html=True)
st.write("<br>", unsafe_allow_html=True)
st.write("<br>", unsafe_allow_html=True)
"""
## Product Review Classifier
The second part is a classifier for product reviews. It uses a dataset of 1'000'000 product reviews from Amazon. Due to 
the large size of the dataset, and computing restraints we can only deploy small classifiers.
"""

st.write("<br>", unsafe_allow_html=True)

product_options = ["BernoulliNB", "LogisticRegression"]
product_method = st.selectbox("Select a method", product_options, 0)
text_product = st.text_area("Enter a review", "would not recommend", key="product")

if st.button("Submit", key="product_submit"):
    # start loading animation
    with st.spinner('Wait for it...'):
        response_product = product_review_classifier.product_review_get_sentiment([text_product], product_method)

    st.write(response_product["stars"])
