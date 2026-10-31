import pandas as pd
import numpy as np

def load_data(features_path, classes_path):
    feat = pd.read_csv(features_path, header=None)
    classes = pd.read_csv(classes_path)

    feat = feat.rename(columns={0: "txId", 1: "time_step"})
    df = feat.merge(classes, on="txId")

    labeled = df[df["class"] != "unknown"].copy()

    labeled["target"] = np.where(labeled["class"] == "1", 1, 0)

    return labeled


def split_data(df):
    train = df[df["time_step"] <= 24]
    val = df[(df["time_step"] > 24) & (df["time_step"] <=34)]
    test = df[df["time_step"] > 34]
    execlude_columns = ["txId", "time_step", "class", "target"]

    X_train, y_train = train.drop(columns=execlude_columns), train["target"]
    X_val, y_val = val.drop(columns=execlude_columns), val["target"]
    X_test, y_test = test.drop(columns=execlude_columns), test["target"]

    return X_train, y_train, X_val, y_val, X_test, y_test