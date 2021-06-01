import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def check_data(X_train, X_test, y_train, y_test):

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

        # vectorizer = CountVectorizer(min_df=3, max_df=0.8, stop_words='english')

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'vect__min_df': [1, 2, 3, 5],
        'vect__max_df': [0.7, 0.8, 0.9, 1.0],
        'vect__stop_words': ['english', None],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
    }

    gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

    gs_clf.fit(X_train[:400], y_train[:400])

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    # Model learning
    # text_clf.fit(X_train, y_train)

    # Checking predictions
    # predicted = text_clf.predict(X_test)

    # correctness = np.mean(predicted == y_test)

    # print()
    # print("EVALUATION")
    # print(correctness)
