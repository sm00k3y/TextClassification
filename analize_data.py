import matplotlib.pyplot as plt
import math
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from const import TOP_X_WORDS



def make_analisys(texts, labels):
    pass


def plot_wordcloud(texts):
    big_texts = ""
    for x in texts:
        big_texts+=x

    wordcloud = WordCloud(width = 1500, height = 1000, random_state=1, background_color='white', collocations=False, stopwords = STOPWORDS).generate(big_texts)

    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud) 
    plt.axis("off")
    plt.show()


def get_classes_ratio(labels):
    zeros = 0
    ones = 0
    twos = 0

    for lab in labels:
        if lab == 0:
            zeros += 1
        elif lab == 1:
            ones += 1
        elif lab == 2:
            twos += 1

    plt.bar(["zero", "one", "two"], [zeros, ones, twos])
    plt.show()

    return zeros, ones, twos
    

def average_review_length(texts):
    min_length = math.inf
    max_length = 0
    sum_of_lengths = 0

    for text in texts:
        text_length = len(text)
        sum_of_lengths += text_length
        if text_length > max_length:
            max_length = text_length
        if text_length < min_length:
            min_length = text_length

    avg = sum_of_lengths / len(texts)

    plt.bar(["Min Length", "Average", "Max Length"], [min_length, avg, max_length])
    plt.show()

    return avg, min_length, max_length


def plot_words_chart(texts):
    cv = CountVectorizer(stop_words='english')
    cv_fit = cv.fit_transform(texts)
    word_list = cv.get_feature_names()
    count_list = np.asarray(cv_fit.sum(axis=0))[0]

    freq_dict = dict(zip(word_list, count_list))

    list_of_words_frequency = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse=True))

    top_x_words = list(list_of_words_frequency.keys())[:TOP_X_WORDS]
    top_x_freq = list(list_of_words_frequency.values())[:TOP_X_WORDS]

    plt.plot(top_x_words, top_x_freq)
    plt.xticks(rotation=90)
    plt.show()


