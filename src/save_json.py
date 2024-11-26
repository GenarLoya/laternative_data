from os import path
import json

current_dir = path.dirname(path.abspath(__file__))
output_path = path.join(current_dir, "../output")


def save_json(dict, name="results.json"):
    with open(path.join(output_path, name), "w") as f:
        json.dump(
            dict,
            f,
            indent=4,
        )


if __name__ == "__main__":
    # * TESTING FUNCTION
    dict_better_acurr = {
        "Naive Bayes": 0.8,
        "Tree Classifier": 0.8,
        "Linear Regression": 0.8,
        "Neuronal Model": {
            "accuracy": 0.8,
            "test_size": 0.25,
            "random_state": 42,
            "activation": "relu",
            "hidden_layer_sizes": (256, 128, 64, 32),
        },
    }

    save_json(dict_better_acurr)
