import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
ORIGIN_DATASET_PATH = os.getenv("ORIGIN_DATASET_PATH")
NEW_DATASET_PATH = os.getenv("NEW_DATASET_PATH")


if __name__ == "__main__":
    df = pd.read_excel(ORIGIN_DATASET_PATH)
    drop_colums: list[str] = list(df.columns[88:210])
    df = df.drop(drop_colums, axis=1)

    df.to_excel(NEW_DATASET_PATH)
