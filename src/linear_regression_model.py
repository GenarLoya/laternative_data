from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from colorama import Fore, Back, Style
from get_df import get_df
from colorama import just_fix_windows_console
from save_json import save_json

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


def execute_linear_regression(df, test_size=0.25, random_state=42):
    print(
        Style.BRIGHT
        + Fore.BLACK
        + "++ 󰙨 Testing for test_size={} && random_state={} ++".format(
            test_size, random_state
        )
        + Style.RESET_ALL
    )

    X = df.drop("is_spam", axis=1)
    y = df["is_spam"]

    X_binary = X.map(lambda x: 1 if x > 0 else 0)

    print("--- Variables ---")
    print("X:")
    print(X_binary.head())
    print("Y:")
    print(y.head())

    print("--- Train Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y, test_size=test_size, random_state=random_state
    )
    print("--- Split ---")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    print("--- Model ---")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("--- Prediction ---")
    print("y_pred:", y_pred)

    # Convertir predicciones continuas a binario
    y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

    score = model.score(X_test, y_test)
    print("--- Score ---")
    print("score:", score)

    print("--- Confusion Matrix ---")

    cm = confusion_matrix(y_test, y_pred_binary)
    try:
        sns.heatmap(cm, annot=True, fmt="d")
        plt.show()
    except Exception as e:
        print(
            Style.BRIGHT
            + Back.RED
            + f"Error: Heatmap can't be showed: {e}"
            + Style.RESET_ALL
        )

    print(Style.BRIGHT + Back.LIGHTMAGENTA_EX + "++ Accuracy ++" + Style.RESET_ALL)
    accuracy = cm.trace() / cm.sum()
    print(accuracy)

    return {
        "accuracy": accuracy,
        "test_size": test_size,
        "random_state": random_state,
    }


def execute_linear_regression_tests_variants(df):
    print(
        Style.BRIGHT
        + Fore.LIGHTMAGENTA_EX
        + "--- Linear Regression 󰙨 Tests ---"
        + Style.RESET_ALL
    )

    acurracies = []

    test_sizes = [0.25, 0.5, 0.75]
    random_states = [42, 43, 44]

    for test_size in test_sizes:
        for random_state in random_states:
            acurracies.append(
                execute_linear_regression(
                    df, test_size=test_size, random_state=random_state
                )
            )

    # * Better accuracy
    save_json(acurracies, name="linear_regression_results.json")
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
    execute_linear_regression_tests_variants(df)
