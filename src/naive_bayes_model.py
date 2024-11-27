from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from colorama import Fore, Back, Style
from get_df import get_df
from colorama import just_fix_windows_console
from save_json import save_json
from matplotlib import pyplot as plt
from save_show_heatmap_as_image import save_show_heatmap_as_image


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

    print("--- Train Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("--- Scale ---")
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print("--- Model ---")
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    print("--- Confusion Matrix ---")
    print(cm)

    try:
        save_show_heatmap_as_image(
            cm,
            folder="naive_bayes",
            filename="conf_test_size_{}_random_state_{}.png".format(
                test_size, random_state
            ),
            title="Confusion Matrix for test_size={} & random_state={}".format(
                test_size, random_state
            ),
        )
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

    test_sizes = [0.10 * i for i in range(1, 8)]

    for test_size in test_sizes:
        acurracies.append(execute_naive_bayes(df, test_size=test_size))

    # * Better accuracy
    save_json(acurracies, "naive_bayes_results.json")

    better_accuracy = None

    for acurracy in acurracies:
        if better_accuracy is None:
            better_accuracy = acurracy
        else:
            if acurracy["accuracy"] > better_accuracy["accuracy"]:
                better_accuracy = acurracy

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
