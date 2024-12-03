from get_df import get_df


def save_df(df):
    df.to_csv("data.csv", index=False)


if __name__ == "__main__":
    df = get_df()
    save_df(df)
