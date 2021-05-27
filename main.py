import data_handler
import analize_data


if __name__ == "__main__":
    texts, labels = data_handler.get_sets()

    analize_data.make_analisys(texts, labels)
