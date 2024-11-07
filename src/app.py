from os import path
from get_df import get_df
from pandas import DataFrame
from colorama import just_fix_windows_console

from naive_bayes_model import execute_naive_bayes
from tree_classifier_model import execute_tree_selector
from linear_regression_model import execute_linear_regression
from neuronal_model import execute_neuronal_model

current_dir = path.dirname(path.abspath(__file__))
output_path = path.join(current_dir, "../output")

if __name__ == "__main__":
    just_fix_windows_console()
    df = get_df()

    # * Executing Models
    naive_bayes_accuracy = execute_naive_bayes(df)
    tree_classifier_accuracy = execute_tree_selector(df)
    linear_regression_accuracy = 0  # execute_linear_regression(df)
    neuronal_model_accuracy = 0  # execute_neuronal_model(df)

    df_acuracy = DataFrame(
        {
            "Model": [
                "Naive Bayes",
                "Tree Classifier",
                "Linear Regression",
                "Neuronal Model",
            ],
            "Accuracy": [
                naive_bayes_accuracy,
                tree_classifier_accuracy,
                linear_regression_accuracy,
                neuronal_model_accuracy,
            ],
        }
    )
    print(df_acuracy)
    df_acuracy.to_csv(path.join(output_path, "accuracy.csv"))
