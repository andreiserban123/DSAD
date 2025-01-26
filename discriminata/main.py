import pandas
import pandas as pd

df_buget = pd.read_csv("data/Buget.csv", index_col=0)
df_pop = pd.read_csv("data/PopulatieLocalitati.csv", index_col=0)

df = df_buget.copy()

venituri = df.columns[1:6]
cheltuieli = df.columns[6:]

df["Venituri"] = df[venituri].sum(axis=1)
df["Cheltuieli"] = df[cheltuieli].sum(axis=1)

df = df[["Localitate", "Venituri", "Cheltuieli"]]

df.to_csv("cerinta1.csv")

cerinta2 = pandas.merge(df_pop, df_buget, left_index=True, right_index=True)

cerinta2 = cerinta2[["Judet", "V1", "V2", "V3", "V4", "V5"]]
cerinta2 = cerinta2.groupby("Judet").sum()

cerinta2 = cerinta2.sort_values(by="V1", ascending=False)


cerinta2.to_csv("cerinta2.csv")
