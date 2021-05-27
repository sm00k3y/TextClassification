import matplotlib.pyplot as plt
import math
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from const import TOP_X_WORDS


def make_analisys(texts, labels):
    zeros, ones, twos = get_classes_ratio(labels)
    print(f"Sum of reviews: {zeros + ones + twos}")
    print(f"Reviews with class 0: {zeros}")
    print(f"Reviews with class 1: {ones}")
    print(f"Reviews with class 2: {twos}")
    print()

    avg, min_length, max_length = average_review_length(texts)
    print(f"Average review length: {math.floor(avg)} letters")
    print(f"Minial review length: {min_length} letters")
    print(f"Maximal review length: {max_length} letters")

    top_words, top_ranks = plot_words_chart(texts)
    print(f"\nTop 20 words without english stop words:")
    for i in range(len(top_words)):
        print(f"{top_words[i]} - {top_ranks[i]}")

    plot_wordcloud(texts)


def plot_wordcloud(texts):
    big_texts = ""
    for x in texts:
        big_texts+=x

    wordcloud = WordCloud(width = 1200, height = 800, random_state=1, background_color='white', collocations=False, stopwords = STOPWORDS).generate(big_texts)

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

    plt.style.use('seaborn')
    plt.bar(["Class zero", "Class one", "Class two"], [zeros, ones, twos])
    plt.title("Number of each class in labels")
    plt.xlabel("Class")
    plt.ylabel("Number of occurences")
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
    plt.title("Length of reviews")
    plt.xlabel("Type of length")
    plt.ylabel("Number of letters")
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
    plt.title("Frequency of top 20 words")
    plt.xlabel("Word")
    plt.ylabel("Number of occurences")
    plt.show()

    return top_x_words, top_x_freq
