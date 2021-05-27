import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from const import TEST_BATCH_SIZE, RANDOM_STATE_SEED


def get_sets():
    dirs = [x[0] for x in os.walk("datasets")]
    dirs = dirs[1:]

    texts = []
    labels = []

    for d in dirs:
        for data_file in os.listdir(d):
            if data_file.startswith("subj"):
                text_file = open(os.path.join(d, data_file), "r")
                texts.extend(text_file.readlines())
            if data_file.startswith("label.3class"):
                label_file = open(os.path.join(d, data_file), "r")
                labels.extend(label_file.readlines())

    int_labels = fix_labes(labels)

    return texts, int_labels


def fix_labes(labels):
    int_labels = []
    for lab in labels:
        int_labels.append(int(lab))
    return int_labels
        

def prepare_data(texts, labels):
    vectorizer = CountVectorizer(min_df=3, max_df=0.8, stop_words='english')
    x = vectorizer.fit_transform(texts).toarray()

    tfidfconverter = TfidfTransformer()
    x = tfidfconverter.fit_transform(x).toarray()

    X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=TEST_BATCH_SIZE, random_state=RANDOM_STATE_SEED)
    
    return X_train, X_test, y_train, y_test