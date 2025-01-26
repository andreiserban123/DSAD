import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.cross_decomposition import CCA

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


# B.
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


print("Datele pentru analiza")
print(df_vot)

df_vot_clean = clean_data(df_vot)

# Separam in categorii
set_barbati = ["Barbati_25-34", "Barbati_35-44", "Barbati_45-64", "Barbati_65_"]
set_femei = ["Femei_18-24", "Femei_35-44", "Femei_45-64", "Femei_65_"]


# Pasul 3: Facem cele doua seturi
X = df_vot_clean[set_barbati]
Y = df_vot_clean[set_femei]

X, Y = X.align(Y, join="inner", axis=0)

# Pas 4: Analiza canonica
cca = CCA()
cca.fit(X, Y)

# Pas 5: Scorurile canonice

X_c, Y_c = cca.transform(X, Y)

df_X_c = pd.DataFrame(X_c, index=X.index, columns=["X_c1", "X_c2"])

df_X_c.to_csv("data/z.csv")
df_Y_c = pd.DataFrame(Y_c, index=Y.index, columns=["Y_c1", "Y_c2"])
df_Y_c.to_csv("data/u.csv")

# Pasul 6: Calculam corelatiile canonice
correlation_canonical = cca.score(X, Y)
with open("data/r.csv", "w") as file:
    file.write(f"Corelatia canonica: {correlation_canonical}")
# Pasul 7: Vizualizarea datelor
plt.figure("Vizualizarea datelor canonice")
plt.scatter(X_c[:, 0], X_c[:, 1], label="Barbati")
plt.scatter(Y_c[:, 0], Y_c[:, 1], label="Femei")
plt.title("Primele doua radacini canonice")
plt.xlabel("Componenta 1")
plt.ylabel("Componenta 2")
plt.show()
