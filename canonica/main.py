import pandas as pd
from pandas import DataFrame


def clean_data(df):
    assert isinstance(df, DataFrame)
    if df.isna().any().any():
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
    return df


df_vot = pd.read_csv("data/Vot.csv", index_col=0)
df_cod = pd.read_csv("data/Coduri_Localitati.csv", index_col=0)


# CERINTA 1
categories = list(df_vot.columns)
categories = categories[2:]

cerinta1 = df_vot.copy()

for category in categories:
    cerinta1[category] = cerinta1[category] * 100 / cerinta1["Votanti_LP"]

cerinta1.drop("Votanti_LP", axis=1, inplace=True)

cerinta1.to_csv("cerinta1.csv", index=True)


# CERINTA 2

df_merged = pd.merge(df_vot, df_cod, left_index=True, right_index=True)

cerinta2 = df_merged[["Judet"] + categories + ["Votanti_LP"]]

cerinta2 = cerinta2.groupby("Judet").sum()

for categ in categories:
    cerinta2[categ] = cerinta2[categ] * 100 / cerinta2["Votanti_LP"]

cerinta2 = cerinta2.drop("Votanti_LP", axis=1)

cerinta2.to_csv("cerinta2.csv")
