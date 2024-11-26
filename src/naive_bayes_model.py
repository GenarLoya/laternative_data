from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from colorama import Fore, Back, Style
from get_df import get_df
from colorama import just_fix_windows_console
from save_json import save_json


def execute_naive_bayes(df, test_size=0.25, random_state=42):
    print(
        Style.BRIGHT
        + Fore.LIGHTMAGENTA_EX
        + "++ 󰙨 Testing for test_size={} && random_state={} ++".format(
            test_size, random_state
        )
        + Style.RESET_ALL
    )
    X = df.drop("is_spam", axis=1)
    y = df["is_spam"]

    print("---Variables---")
    print("X:")
    print(X)
    print("Y:")
    print(y)

    print("--- Train Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("--- Scale ---")
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print("++ X_train ++")
    print(X_train)
    print("++ X_test ++")
    print(X_test)

    print("--- Model ---")
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

    return {
        "accuracy": accuracy,
        "test_size": test_size,
        "random_state": random_state,
    }


def execute_naive_bayes_tests_variants(df):
    print(
        Style.BRIGHT
        + Fore.LIGHTMAGENTA_EX
        + "--- Naive Bayes 󰙨 Tests ---"
        + Style.RESET_ALL
    )

    acurracies = []

    acurracies.append(execute_naive_bayes(df, test_size=0.25, random_state=42))
    acurracies.append(execute_naive_bayes(df, test_size=0.5, random_state=43))
    acurracies.append(execute_naive_bayes(df, test_size=0.75, random_state=44))

    # * Better accuracy
    save_json(acurracies, "naive_bayes_results.json")

    better_accuracy = None

    for acurracy in acurracies:
        if better_accuracy is None:
            better_accuracy = acurracy
        else:
            if acurracy["accuracy"] > better_accuracy["accuracy"]:
                better_accuracy = acurracy

    print("--- Better Accuracy ---")
    print(better_accuracy)

    return better_accuracy


if __name__ == "__main__":
    just_fix_windows_console()

    df = get_df()
    execute_naive_bayes_tests_variants(df)
