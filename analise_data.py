import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


def plot_wordcloud(texts):
    big_texts = ""
    for x in texts:
        big_texts+=x

    wordcloud = WordCloud(width = 1500, height = 1000, random_state=1, background_color='white', collocations=False, stopwords = STOPWORDS).generate(big_texts)

    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud) 
    plt.axis("off")
    plt.show()
