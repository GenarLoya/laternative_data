from os import path
from get_df import get_df
from pandas import DataFrame
from colorama import just_fix_windows_console

from naive_bayes_model import execute_naive_bayes_tests_variants
from tree_classifier_model import execute_tree_selector_tests_variants
from linear_regression_model import execute_linear_regression_tests_variants
from neuronal_model import execute_neuronal_model_tests_variants
from save_json import save_json

current_dir = path.dirname(path.abspath(__file__))
output_path = path.join(current_dir, "../output")

if __name__ == "__main__":
    just_fix_windows_console()
    df = get_df()

    # * Executing Models
    naive_bayes_accuracy = execute_naive_bayes_tests_variants(df)
    tree_classifier_accuracy = execute_tree_selector_tests_variants(df)
    linear_regression_accuracy = execute_linear_regression_tests_variants(df)
    neuronal_model_accuracy = execute_neuronal_model_tests_variants(df)

    dict_better_acurr = {
        "Naive Bayes": naive_bayes_accuracy,
        "Tree Classifier": tree_classifier_accuracy,
        "Linear Regression": linear_regression_accuracy,
        "Neuronal Model": neuronal_model_accuracy,
    }

    # * Save json file
    save_json(dict_better_acurr, name="bettar_acuracy.json")
