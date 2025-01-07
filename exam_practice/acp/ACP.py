import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Citirea datelor din fișiere CSV
df_mortalitate = pd.read_csv("Mortalitate.csv", index_col=0)
df_coduri = pd.read_csv("CoduriTariExtins.csv", index_col=0)


# Funcție pentru tratarea valorilor lipsă (NA)
def nan_replace(t):
    assert isinstance(t, pd.DataFrame)
    for i in t.columns:
        if any(t[i].isna()):
            # Pentru variabile numerice, înlocuim NA cu media
            t[i] = t[i].fillna(t[i].mean())
        else:
            # Pentru variabile categorice, înlocuim NA cu modul (valoarea cea mai frecventă)
            t[i] = t[i].fillna(t[i].mode()[0])


# Aplicăm tratarea valorilor lipsă pe ambele seturi de date
nan_replace(df_mortalitate)
nan_replace(df_coduri)

# Selectăm toate coloanele cu date, exceptând prima coloană
set_date = list(df_mortalitate.columns)
x = df_mortalitate[set_date].values

# Standardizarea datelor (media 0, deviație standard 1)
# Acest pas este crucial pentru ACP deoarece variabilele au unități de măsură diferite
scaler = StandardScaler()
date_standardizate = scaler.fit_transform(df_mortalitate)

# Aplicarea ACP pe datele standardizate
modelACP = PCA()
# C conține coordonatele observațiilor în spațiul componentelor principale
C = modelACP.fit_transform(date_standardizate)

# Calculul varianței explicate de fiecare componentă principală
varianta = modelACP.explained_variance_


# Calculul procentului de varianță explicată
procent_varianta = modelACP.explained_variance_ratio_ * 100
print("\nProcentul de varianță explicată de fiecare componentă:")
for i, procent in enumerate(procent_varianta):
    print(f"C{i+1}: {procent:.2f}%")

# Calculul scorurilor (coordonatele standardizate ale observațiilor în spațiul CP)
scoruri = C / np.sqrt(varianta)
etichetari = ["C" + str(i + 1) for i in range(len(varianta))]
df_scoruri = pd.DataFrame(data=scoruri, index=df_mortalitate.index, columns=etichetari)

# Vizualizarea scorurilor în primele două componente principale
plt.figure("Plot scoruri in cele 2 axe principale", figsize=(9, 9))
plt.subplot(1, 1, 1)
plt.xlabel("C1")
plt.ylabel("C2")
plt.title("Plot scoruri in cele doua axe principale")
plt.scatter(df_scoruri["C1"], df_scoruri["C2"], color="r")
# Adăugarea etichetelor pentru fiecare punct
for index, (x, y) in df_scoruri[["C1", "C2"]].iterrows():
    plt.text(x, y, index)

# Calculul corelațiilor dintre variabilele inițiale și componentele principale
r_x_c = np.corrcoef(date_standardizate, C, rowvar=False)[
    : len(varianta), len(varianta) :
]
eticheta_corel = ["C" + str(i + 1) for i in range(len(varianta))]
df_corelatii = pd.DataFrame(
    data=r_x_c, index=df_mortalitate.columns, columns=[eticheta_corel]
)

# Vizualizarea corelațiilor prin heatmap
plt.figure("Corelograma", figsize=(9, 9))
plt.subplot(1, 1, 1)
sb.heatmap(data=r_x_c, vmin=-1, vmax=1, cmap="bwr", annot=True)
plt.title("Corelograma factoriala")

# Calculul contribuțiilor (cât contribuie fiecare observație la formarea CP)
C_patrat = C * C
contributii = C_patrat / np.sum(C_patrat, axis=0)
df_contributii = pd.DataFrame(
    data=contributii, index=df_mortalitate.index, columns=eticheta_corel
)

# Calculul cosinusurilor (calitatea reprezentării observațiilor în spațiul CP)
cosinusuri = np.transpose(C_patrat.T / np.sum(C_patrat, axis=1))
df_cosinusuri = pd.DataFrame(
    data=cosinusuri, index=df_mortalitate.index, columns=eticheta_corel
)

# Calculul comunalităților (proporția din varianța variabilelor explicată de CP)
comunalitati = np.cumsum(r_x_c * r_x_c, axis=1)
df_comunalitati = pd.DataFrame(
    data=comunalitati, index=df_mortalitate.columns, columns=eticheta_corel
)

# Vizualizarea comunalităților prin heatmap
plt.figure("Corelegrama comunalitati", figsize=(9, 9))
plt.subplot(1, 1, 1)
sb.heatmap(data=df_comunalitati, vmin=-1, vmax=1, cmap="bwr", annot=True)
plt.title("Corelograma comunalitati")
plt.show()
