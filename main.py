import load_data
import analise_data


if __name__ == "__main__":
    texts, labels = load_data.get_sets()
    analise_data.plot_wordcloud(texts)