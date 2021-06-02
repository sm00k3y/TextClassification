import data_handler
import analize_data
import ml_bayes
import ml_svg
import sys
import os


def run():
    # Pobranie danych z datasetow
    texts, labels = data_handler.get_sets()

    # Wykonanie calej analizy danych
    # analize_data.make_analisys(texts, labels)

    # Get test and train baches and pass them to the AI
    X_train, X_test, y_train, y_test = data_handler.prepare_data(texts, labels)

    # ml_bayes.evaluate_model(X_train, X_test, y_train, y_test)
    ml_svg.evaluate_model(X_train, X_test, y_train, y_test)



if __name__ == "__main__":

    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))

    run()
