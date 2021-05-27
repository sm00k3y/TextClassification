import data_handler
import analize_data
import sys
import os


def run():
    # Pobranie danych z datasetow
    texts, labels = data_handler.get_sets()

    # Wykonanie calej analizy danych
    analize_data.make_analisys(texts, labels)


if __name__ == "__main__":

    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))

    run()
