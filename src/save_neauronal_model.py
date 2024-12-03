from neuronal_model import execute_neuronal_model
from get_df import get_df

# This execution is only used when final model is selected

if __name__ == "__main__":
    df = get_df()

    execute_neuronal_model(
        df,
        test_size=0.25,
        random_state=42,
        activation="relu",
        hidden_layer_sizes=(256, 128, 64, 32),
    )
