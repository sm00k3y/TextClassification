import data_handler
import analize_data


if __name__ == "__main__":
    texts, labels = data_handler.get_sets()
    # analize_data.plot_wordcloud(texts)
    # analize_data.plot_words_chart(texts)
    # analize_data.get_classes_ratio(labels)
    analize_data.average_review_length(texts)
