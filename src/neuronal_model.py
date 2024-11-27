# Neuronal
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from colorama import Fore, Back, Style
from get_df import get_df
from colorama import just_fix_windows_console

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
from save_json import save_json
from save_show_heatmap_as_image import save_show_heatmap_as_image


def execute_neuronal_model(
    df,
    test_size=0.25,
    random_state=42,
    activation="relu",
    hidden_layer_sizes=(256, 128, 64, 32),
    max_iter=1000,
):
    print(
        Style.BRIGHT
        + Fore.BLACK
        + Back.MAGENTA
        + "++ 󰙨 Testing for test_size={} && random_state={} && activation={} && hidden_layer_sizes={} && max_iter={} ++".format(
            test_size, random_state, activation, hidden_layer_sizes, max_iter
        )
        + Style.RESET_ALL
    )
    X = df.drop("is_spam", axis=1)
    y = df["is_spam"]

    print("--- Variables processing ---")
    X_binary = X.map(lambda x: 1 if x > 0 else 0)
    # print("---Variables---")
    # print("X:")
    # print(X_binary)
    # print("Y:")
    # print(y)

    print("--- Train Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y, test_size=test_size, random_state=random_state
    )

    print("--- Scale ---")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("--- Model ---")
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=1,
        verbose=True,
        activation=activation,
    )
    mlp.fit(X_train, y_train)

    print("--- Confusion Matrix ---")
    y_pred = mlp.predict(X_test)
    # print(y_pred)

    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    try:
        save_show_heatmap_as_image(
            cm,
            folder="neuronal_model",
            filename="conf_test_size_{}_random_state_{}_activation_{}_hidden_layer_sizes_{}.png".format(
                test_size, random_state, activation, hidden_layer_sizes
            ),
            title="Confusion Matrix for test_size={} & random_state={} and activation={} & hidden_layer_sizes={}".format(
                test_size, random_state, activation, hidden_layer_sizes
            ),
        )
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
        "activation": activation,
        "hidden_layer_sizes": hidden_layer_sizes,
    }


def execute_neuronal_model_tests_variants(df):
    print(
        Style.BRIGHT
        + Fore.LIGHTMAGENTA_EX
        + "--- Neuronal Model 󰙨 Tests ---"
        + Style.RESET_ALL
    )

    acurracies = []

    layer_variants = [
        (256, 128, 64, 32),
        (128, 64, 32),
        (64, 32),
        (32),
    ]

    activation_variants = ["relu", "tanh", "identity", "logistic"]

    for layer in layer_variants:
        for activation in activation_variants:
            acurr = execute_neuronal_model(
                df,
                test_size=0.25,
                random_state=42,
                activation=activation,
                hidden_layer_sizes=layer,
            )

            acurracies.append(acurr)

        # print(acurracies)

    # * Better accuracy
    save_json(acurracies, name="neuronal_model_results.json")

    better_accuracy = None

    for acurracy in acurracies:
        if better_accuracy is None:
            better_accuracy = acurracy
        else:
            if acurracy["accuracy"] > better_accuracy["accuracy"]:
                better_accuracy = acurracy

    print("--- Better Accuracy ---")
    print(better_accuracy)


if __name__ == "__main__":
    just_fix_windows_console()

    df = get_df()
    execute_neuronal_model_tests_variants(df)
