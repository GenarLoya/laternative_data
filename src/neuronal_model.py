#Neuronal
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


def execute_neuronal_model(df):
    X = df.drop("is_spam", axis=1)
    y = df["is_spam"]

    X_binary = X.map(lambda x: 1 if x > 0 else 0)
    print("---Variables---")
    print("X:")
    print(X_binary)
    print("Y:")
    print(y)


    print("--- Train Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y, test_size=0.25, random_state=True
    )
    
    print("--- Scale ---")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("--- Model ---")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32),
                        max_iter=1000, random_state=1, verbose=True, activation='relu')
    mlp.fit(X_train, y_train)

    print("--- Confusion Matrix ---")
    y_pred = mlp.predict(X_test)
    #print(y_pred)

    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    try:
        sns.heatmap(cm, annot=True, fmt="d")
    except Exception as e:
        print(
            Style.BRIGHT + Back.RED + f"Error: Heatmap can't be showed: {e}" + Style.RESET_ALL
        )

    print(Style.BRIGHT + Back.LIGHTMAGENTA_EX + "++ Accuracy ++" + Style.RESET_ALL)
    accuracy = cm.trace() / cm.sum()
    print(accuracy)

    return accuracy

print("OSAOD")
if __name__ == "__main__":
    just_fix_windows_console()

    df = get_df()
    execute_neuronal_model(df)