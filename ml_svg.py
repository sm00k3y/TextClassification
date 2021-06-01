import numpy as np



def check_data_svm(X_train, X_test, y_train, y_test):
    print(np.shape(X_train))
    print(X_train[0])

    # Model learning
    # clf = MultinomialNB().fit(X_train, y_train)

    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
    clf.fit(X_train, y_train)

    # Checking predictions
    predicted = clf.predict(X_test)

    correctness = np.mean(predicted == y_test)

    print()
    print("EVALUATION")
    print(correctness)



def check_data_svm_params(X_train, X_test, y_train, y_test):

    parameters = {
        'alpha': (1e-2, 1e-3),
    }


    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
    clf.fit(X_train, y_train)

    gs_clf = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)
    gs_clf = gs_clf.fit(X_train[:400], y_train[:400])

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


    # Checking predictions
    # predicted = clf.predict(X_test)

    # correctness = np.mean(predicted == y_test)

    # print()
    # print("EVALUATION")
    # print(correctness)