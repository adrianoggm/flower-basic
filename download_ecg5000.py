# download_ecg5000.py
import os

import kagglehub
import pandas as pd


def fetch_and_prepare(output_path="data/ecg5000.csv"):
    # 1) download the Kaggle dataset to a temp folder
    tmp_dir = kagglehub.dataset_download("salsabilahmid/ecg50000")
    # 2) the ECG5000 files are usually in .tsv or .txt—find them:
    #    for simplicity, assume there's ECG5000_TRAIN.tsv & ECG5000_TEST.tsv
    train_tsv = os.path.join(tmp_dir, "ECG5000_TRAIN.tsv")
    test_tsv = os.path.join(tmp_dir, "ECG5000_TEST.tsv")

    # 3) read, concat, and rename columns
    df_train = pd.read_csv(train_tsv, sep=r"\s+", header=None)
    df_test = pd.read_csv(test_tsv, sep=r"\s+", header=None)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # first column is label, the next 140 are features
    n_features = df.shape[1] - 1
    df.columns = ["label"] + [f"t{i}" for i in range(n_features)]

    # 4) ensure data/ exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✔️ Saved combined ECG5000 to {output_path}, shape={df.shape}")


if __name__ == "__main__":
    fetch_and_prepare()
