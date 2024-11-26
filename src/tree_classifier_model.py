from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
from colorama import Back, Style
from get_df import get_df
from colorama import just_fix_windows_console, Fore
from save_json import save_json


def execute_tree_selector(df, test_size=0.25, random_state=42):
    print(
        Back.GREEN
        + Fore.BLACK
        + Style.BRIGHT
        + "++ Test for test_size={} && random_state={} ++".format(
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
    X_train, X_test, Y_train, Y_Test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("++ X_train ++")
    print(X_train)
    print("++ X_test ++")
    print(X_test)

    print("--- Model ---")
    clf = DecisionTreeClassifier()

    # Training
    tree = clf.fit(X_train, Y_train)
    print("----PREDICTION-----")
    y_pred = tree.predict(X_test)
    print(y_pred)

    print("--- Graph ---")
    try:
        plt.figure(figsize=(25, 20))
        plot_tree(
            tree,
            feature_names=list(X.columns.values),
            class_names=["Spam", "Ham"],
            filled=True,
        )
        plt.show()
    except:
        print(Style.BRIGHT + Back.RED + "Error: Tree can't be showed" + Style.RESET_ALL)

    print("--- Confusion Matrix ---")

    cm = confusion_matrix(Y_Test, y_pred)
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


def execute_tree_selector_tests_variants(df):
    print(
        Style.BRIGHT
        + Fore.LIGHTMAGENTA_EX
        + "--- Tree Classifier 󰙨 Tests ---"
        + Style.RESET_ALL
    )

    acurracies = []

    test_sizes = [0.25, 0.5, 0.75]
    random_states = [42, 43, 44]

    for test_size in test_sizes:
        for random_state in random_states:
            acurracies.append(
                execute_tree_selector(
                    df, test_size=test_size, random_state=random_state
                )
            )

    # * Better accuracy
    save_json(acurracies, name="tree_classifier_results.json")
    better_accuracy = acurracies.sort(key=lambda x: x["accuracy"], reverse=True)[0]
    print("--- Better Accuracy ---")
    print(better_accuracy)

    return better_accuracy


if __name__ == "__main__":
    just_fix_windows_console()

    df = get_df()
    execute_tree_selector_tests_variants(df)
