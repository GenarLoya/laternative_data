import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from colorama import Fore, Back, Style

current_dir = os.path.dirname(os.path.abspath(__file__))

route_ham = "../data/ham"
route_spam = "../data/spam"

relative_spam_path = os.path.join(current_dir, route_spam)
relative_ham_path = os.path.join(current_dir, route_ham)


def get_df():
    print(
        Back.GREEN
        + Fore.LIGHTMAGENTA_EX
        + Fore.WHITE
        + "Getting processed data"
        + Style.RESET_ALL
    )
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

    print("X...")
    print(X)

    print("Getting vocabulary...")
    vocabulary = vectorizer.get_feature_names_out()
    print("Vocabulary...")
    for i in range(len(vocabulary)):
        print(vocabulary[i])

    print("Transforming documents...")
    df = pd.DataFrame(X.toarray(), columns=vocabulary)

    print("Dataframe...")
    print(df)

    print("Labels...")
    df["is_spam"] = labels
    print(df)

    return df


if __name__ == "__main__":
    get_df()
