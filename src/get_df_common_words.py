from get_df import get_df
from save_json import save_json


def get_top_words_by_label(df, label_column="is_spam", top_n=300):
    """
    Obtiene las palabras más comunes para cada categoría (spam y ham) en el DataFrame.

    Args:
        df (pd.DataFrame): DataFrame que contiene las palabras como columnas y una columna `is_spam` indicando la categoría.
        label_column (str): Nombre de la columna que indica si un correo es spam o ham.
        top_n (int): Número de palabras más comunes a retornar.

    Returns:
        dict: Un diccionario con dos claves ('spam', 'ham'), cada una con las palabras más comunes y sus frecuencias.
    """
    result = {}
    # Procesar para spam y ham
    for label in [0, 1]:  # 0: ham, 1: spam
        label_name = "ham" if label == 0 else "spam"
        print(f"Processing top {top_n} words for {label_name}...")

        # Filtrar los correos por la etiqueta
        filtered_df = df[df[label_column] == label]

        # Sumar frecuencias por columna (palabra)
        word_counts = (
            filtered_df.drop(columns=[label_column]).sum().sort_values(ascending=False)
        )

        # Seleccionar las palabras más comunes
        top_words = word_counts.head(top_n)

        # Convertir a diccionario
        result[label_name] = top_words.to_dict()

    return result


if __name__ == "__main__":
    # Generar el DataFrame
    df = get_df()

    # Obtener las 300 palabras más comunes
    top_words = get_top_words_by_label(df, top_n=300)

    # Identificar palabras comunes entre spam y ham
    common_words = set(top_words["spam"].keys()) & set(top_words["ham"].keys())

    # Eliminar palabras comunes de ambas categorías
    top_words["spam"] = {
        word: freq
        for word, freq in top_words["spam"].items()
        if word not in common_words
    }
    top_words["ham"] = {
        word: freq
        for word, freq in top_words["ham"].items()
        if word not in common_words
    }

    print("\nFiltered top words for spam:")
    print(top_words["spam"])

    print("\nFiltered top words for ham:")
    print(top_words["ham"])

    # Guardar los resultados en un archivo JSON
    save_json(
        {
            "top_spam_words": top_words["spam"],
            "top_ham_words": top_words["ham"],
        },
        name="top_words.json",
    )
