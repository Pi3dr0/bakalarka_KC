import pandas as pd


if __name__ == "__main__":
    dataset_path: str = "../datasets/All_data_preprocessed.xlsx"
    new_dataset_path: str = "./data/raw/dataset.xlsx"

    df = pd.read_excel(dataset_path)
    drop_colums: list[str] = list(df.columns[88:210])
    df = df.drop(drop_colums, axis=1)

    df.to_excel(new_dataset_path)
