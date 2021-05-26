import os


def get_sets():
    dirs = [x[0] for x in os.walk("datasets")]
    dirs = dirs[1:]

    texts = []
    labels = []

    for d in dirs:
        for data_file in os.listdir(d):
            if data_file.startswith("subj"):
                # print("Data:", data_file)
                text_file = open(os.path.join(d, data_file), "r")
                texts.extend(text_file.readlines())
            if data_file.startswith("label.3class"):
                # print("Label:", data_file)
                label_file = open(os.path.join(d, data_file), "r")
                labels.extend(label_file.readlines())

    return texts, labels
        
