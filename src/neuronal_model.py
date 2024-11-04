from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from colorama import Fore, Back, Style
from get_df import get_df
from colorama import just_fix_windows_console


def execute_neuronal_model(df):
    X = df.drop("is_spam", axis=1)
    y = df["is_spam"]

    print("---Variables---")
    print("X:")
    print(X)
    print("Y:")
    print(y)

    print("--- Train Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=True
    )

    print(Style.BRIGHT + Back.LIGHTMAGENTA_EX + "++ Accuracy ++" + Style.RESET_ALL)
    # TODO: Implement Real Accuracy here
    accuracy = 0.0
    print(accuracy)

    return accuracy


if __name__ == "__main__":
    just_fix_windows_console()

    df = get_df()
    execute_neuronal_model(df)
