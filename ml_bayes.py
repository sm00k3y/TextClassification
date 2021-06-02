import math
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def evaluate_model(X_train, X_test, y_train, y_test):
    # Model pipeline and parameters
    text_clf = Pipeline([
        # ('vect', CountVectorizer(max_df=1.0, max_features=5000, min_df=0.01, ngram_range=(1,2))),
        ('vect', CountVectorizer(max_df=0.8, max_features=5000, min_df=0.01, ngram_range=(1,2))),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', MultinomialNB(alpha=0.01)),
    ])

    # Training the model
    text_clf.fit(X_train, y_train)

    # Checking predictions
    predicted = text_clf.predict(X_test)

    correctness = np.mean(predicted == y_test)

    print()
    print("BAYES EVALUATION:", correctness)


def get_best_params(X_train, X_test, y_train, y_test):
    print(np.shape(X_train))

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    parameters = {
        'vect__max_features': [None, 500, 1_000, 3_000, 5_000],
        'vect__ngram_range': [(1, 1), (1, 2)],
        'vect__min_df': [1, 0.001, 0.01, 0.1, 0.2],
        'vect__max_df': [0.7, 0.8, 0.9, 1.0],
        'vect__stop_words': ['english', None],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1.0, 0.1, 0.01, 0.001, 0),
    }

    gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

    # Take 1/4 of the data to speed up the process of searching for best params
    search_len = math.floor(len(X_train) / 4)
    gs_clf.fit(X_train[:search_len], y_train[:search_len])

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
