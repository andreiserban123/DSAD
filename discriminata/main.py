import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, accuracy_score

df_buget = pd.read_csv("data/Buget.csv", index_col=0)
df_pop = pd.read_csv("data/PopulatieLocalitati.csv", index_col=0)

df = df_buget.copy()

venituri = df.columns[1:6]
cheltuieli = df.columns[6:]

df["Venituri"] = df[venituri].sum(axis=1)
df["Cheltuieli"] = df[cheltuieli].sum(axis=1)

df = df[["Localitate", "Venituri", "Cheltuieli"]]

df.to_csv("cerinta1.csv")

cerinta2 = pd.merge(df_pop, df_buget, left_index=True, right_index=True)
cerinta2 = cerinta2[["Judet", "V1", "V2", "V3", "V4", "V5", "Populatie"]]
cerinta2 = cerinta2.groupby("Judet").sum()
cerinta2 = cerinta2.div(cerinta2["Populatie"], axis=0)  # div by col
cerinta2 = cerinta2.sort_values(by="V1", ascending=False)
cerinta2 = cerinta2.drop("Populatie", axis=1)  # drop col


cerinta2.to_csv("cerinta2.csv")

# Analiza

df_pacienti = pd.read_csv("data/Pacienti.csv", index_col=0)
df_apply = pd.read_csv("data/Pacienti_apply.csv", index_col=0)


def clean_data(df):
    assert isinstance(df, pd.DataFrame)
    if df.isna().any().any():
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

    return df


df_pacienti = clean_data(df_pacienti)
df_apply = clean_data(df_apply)

# Separam variabil dependenta de cele indendente
X = df_pacienti.drop(columns=["DECISION"])
Y = df_pacienti["DECISION"]


# Impartim in date de test si de antrenament
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)


# Crearea si antrenarea modelului
lda = LDA()
lda.fit(X_train, Y_train)


# Predictia pentru modelul LDA
Y_pred = lda.predict(X_test)
df_predict = pd.DataFrame({"Real": Y_test, "Predicted": Y_pred})
df_predict.to_csv("data/predict.csv")


# Evaluarea performantei modelului

# Matricea de confuzie
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Matricea de confuzie:")
print(conf_matrix)
# Acuratetea globala
global_accuracy = accuracy_score(Y_test, Y_pred)
print("Acuratetea globala: ", global_accuracy)

# Acuratetea medie
accuracy_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
accuracy_mean = np.mean(accuracy_per_class)
print("Acuratetea medie: ", accuracy_mean)

# Graficul distributiei

# Transformam datele
X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)

num_axes = X_train_lda.shape[1]

if num_axes < 2:
    print(
        "Numărul de axe discriminante este mai mic decât 2. Graficul nu poate fi generat."
    )
else:
    plt.figure(figsize=(10, 6))
    for label, marker, color in zip(
        np.unique(Y_train), ("o", "x", "s"), ("blue", "red", "green")
    ):
        label_indices = np.where(Y_train == label)[0]
        plt.scatter(
            X_train_lda[label_indices, 0],
            X_train_lda[label_indices, 1],
            label=f"Clasa {label}",
            marker=marker,
            color=color,
        )

    plt.title("Distribuțiile pe axele discriminante")
    plt.xlabel("Axa discriminantă 1")
    plt.ylabel("Axa discriminantă 2")
    plt.legend(title="Clase")
    plt.grid()
    plt.tight_layout()
    plt.show()
