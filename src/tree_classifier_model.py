from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
from colorama import Back, Style
from get_df import get_df
from colorama import just_fix_windows_console


def execute_tree_selector(df):
    print(Back.GREEN + Style.BRIGHT + "Tree Classifier" + Style.RESET_ALL)

    X = df.drop("is_spam", axis=1)
    y = df["is_spam"]

    print("---Variables---")
    print("X:")
    print(X)
    print("Y:")
    print(y)

    print("--- Train Test Split ---")
    X_train, X_test, Y_train, Y_Test = train_test_split(
        X, y, test_size=0.25, random_state=True
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

    return accuracy


if __name__ == "__main__":
    just_fix_windows_console()

    df = get_df()
    execute_tree_selector(df)
