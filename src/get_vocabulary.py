import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from colorama import Fore, Back, Style
from load_model import load_model

current_dir = os.path.dirname(os.path.abspath(__file__))

route_ham = "../data/ham"
route_spam = "../data/spam"

relative_spam_path = os.path.join(current_dir, route_spam)
relative_ham_path = os.path.join(current_dir, route_ham)


def get_vocabulary():
    print(Back.GREEN + Fore.BLACK + "Getting processed data" + Style.RESET_ALL)
    print("Route ham:", relative_ham_path)
    ham_files = glob.glob(os.path.join(relative_ham_path, "*.txt"))
    print("Processing ham files...")
    ham_texts = [open(file, "r", encoding="utf-8").read() for file in ham_files]
    ham_labels = [0] * len(ham_texts)  # 0 indica "no es spam"

    print("Route spam:", relative_spam_path)
    spam_files = glob.glob(os.path.join(relative_spam_path, "*.txt"))
    print("Processing spam files...")
    spam_texts = [
        open(file, "r", encoding="utf-8", errors="ignore").read() for file in spam_files
    ]
    spam_labels = [1] * len(spam_texts)  # 1 indica "es spam"

    documentos = ham_texts + spam_texts
    labels = ham_labels + spam_labels

    print("Processing documents...")
    vectorizer = CountVectorizer()
    print("Fitting vectorizer...")
    X = vectorizer.fit_transform(documentos)

    # print("X...")
    # print(X)

    print("Getting vocabulary...")
    vocabulary = vectorizer.get_feature_names_out()

    return vocabulary


def get_vocabulary_count(email, vocabulary):

    vectorizer = CountVectorizer(vocabulary=vocabulary)

    X = vectorizer.fit_transform([email])

    counts = X.toarray().flatten()

    return counts


## Example loading model with custom input
if __name__ == "__main__":
    vocabulary = get_vocabulary()
    print(vocabulary)

    email = """
Subject: buy cheap prescription drugs online dd
top rated online store .
hot new - levitra / lipitor / nexium
weekly speciasls on all our drugs .
- zocor
- soma
- ambien
- phentermine
- vlagra
- discount generic ' s on all
- more
next day discrete shipping on all products !
http : / / www . rxstoreusa . biz / shopping
please , i wish to receive no more discounts on valuable items .
http : / / www . rxstoreusa . biz / a . html
jet
djjdnj 33 xks npvjkps ekhvhdqkxhm xvgwk
cpjtrsbqgogmjnyi
uknuilrj
moqwrcaigwvvfpsljzycp
k p
e p
gp c j
    """
    counts = get_vocabulary_count(email, vocabulary)

    model = load_model()
    for i in range(len(vocabulary)):
        if counts[i] != 0:
            print(vocabulary[i], counts[i])
    y_pred = model.predict([counts])
    print(y_pred)
