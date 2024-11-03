from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from colorama import Fore, Back, Style
from get_df import get_df
from colorama import just_fix_windows_console


def execute_naive_bayes(df):
    X = df.drop("is_spam", axis=1)
    y = df["is_spam"]

    print("---Variables---")
    print("X:")
    print(X)
    print("Y:")
    print(y)

    print("--- Train Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=True
    )

    print("--- Scale ---")
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print("++ X_train ++")
    print(X_train)
    print("++ X_test ++")
    print(X_test)

    print("--- Naive Bayes ---")
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("--- Confusion Matrix ---")

    cm = confusion_matrix(y_test, y_pred)
    try:
        sns.heatmap(cm, annot=True)
    except:
        print(
            Style.BRIGHT + Back.RED + "Error: Heatmap can't be showed" + Style.RESET_ALL
        )

    print(Style.BRIGHT + Back.LIGHTMAGENTA_EX + "++ Accuracy ++" + Style.RESET_ALL)
    accuracy = cm.trace() / cm.sum()
    print(accuracy)


if __name__ == "__main__":
    just_fix_windows_console()

    df = get_df()
    execute_naive_bayes(df)
