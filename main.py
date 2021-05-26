import data_handler
import analise_data


if __name__ == "__main__":
    texts, labels = data_handler.get_sets()
    analise_data.plot_wordcloud(texts)