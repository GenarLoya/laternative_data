from get_df import get_df
from naive_bayes_model import execute_naive_bayes
from tree_classifier_model import execute_tree_selector
from colorama import just_fix_windows_console

if __name__ == "__main__":
    just_fix_windows_console()
    df = get_df()

    # * Executing Models
    execute_naive_bayes(df)
    execute_tree_selector(df)
