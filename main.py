import pandas as pd


def main():
    data = pd.read_csv("data/raw/claims.csv")

    print(data.dtypes)


if __name__ == "__main__":
    main()
