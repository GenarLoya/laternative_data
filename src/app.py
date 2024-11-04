from get_df import get_df
from pandas import DataFrame
from naive_bayes_model import execute_naive_bayes
from tree_classifier_model import execute_tree_selector
from colorama import just_fix_windows_console

if __name__ == "__main__":
    just_fix_windows_console()
    df = get_df()

    # * Executing Models
    naive_bayes_accuracy = execute_naive_bayes(df)
    tree_classifier_accuracy = execute_tree_selector(df)

    df_acuracy = DataFrame(
        {
            "Model": ["Naive Bayes", "Tree Classifier"],
            "Accuracy": [naive_bayes_accuracy, tree_classifier_accuracy],
        }
    )
    print(df_acuracy)
