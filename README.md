# API repo for https://api.jakob-fender.com
The api has been created with Flask. It has the following endpoints:
## movieReviews
You can choose between the following models:
- default: If left empty, the default model will be used. This is a neural network with ~99.6% accuracy
- `custom`: A custom classifier with an accuracy of ~80%
- `BernoulliNB`: 77.31%
- `ComplementNB`: 78.15%
- `MultinomialNB`: 78.16%
- `KNeighborsClassifier`: 78.10%
- `DecisionTreeClassifier`: 73.08%
- `RandomForestClassifier`: 77.53%
- `LogisticRegression`: 80.60%
- `MLPClassifier`: 80.57%
- `AdaBoostClassifier`: 80.53%

The default and custom model are also giving a confidence score. The confidence score is a value between 0 and 1. The closer the value is to 1, the more confident the model is that the sentiment is correct. 
### /movieReviews/getSentiment
This endpoint takes a movie review as input and returns the sentiment of the review. The sentiment is either `positive` or `negative`. The endpoint is a POST endpoint and accepts the following parameters:
#### Example request
```json
{
    "text": "I really liked this movie. It was great!",
    "model": "custom"
}
```
#### Example response
```json
{
    "sentiment": "positive",
    "confidence": 0.98
}
```
### /movieReviews/getSentiments
This endpoint takes a movie review as input and returns the sentiment of the review. The sentiment is either `positive` or `negative`. The endpoint is a POST endpoint and accepts the following parameters:
#### Example request
```json
{
    "texts": ["I really liked this movie. It was great!", "I really hated this movie. It was terrible!", "I really liked this movie. It was great!"]],
    "model": "custom"
}
```
#### Example response
```json
[
    {"sentiment": "positive", "confidence": 0.98},
    {"sentiment": "negative", "confidence": 0.95},
    {"sentiment": "positive", "confidence": 0.98}
]
```
