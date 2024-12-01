# Importăm bibliotecile necesare
import matplotlib.pyplot as plt  # pentru vizualizări
import numpy as np              # pentru calcule numerice
import pandas as pd            # pentru manipularea datelor
import seaborn as sb          # pentru vizualizări statistice
from sklearn.decomposition import PCA  # pentru Analiza Componentelor Principale
from sklearn.preprocessing import StandardScaler  # pentru standardizarea datelor

# Citim datele din fișiere CSV
df_mortalitate = pd.read_csv('Mortalitate.csv', index_col=0)
df_coduri = pd.read_csv('CoduriTariExtins.csv', index_col=0)

# Definim o funcție pentru înlocuirea valorilor lipsă
def nan_replace(t):
    assert isinstance(t, pd.DataFrame)  # verificăm dacă input-ul este DataFrame
    for i in t.columns:
        if any(t[i].isna()):  # dacă există valori lipsă
            t[i].fillna(t[i].mean(), inplace=True)  # le înlocuim cu media pentru date numerice
        else:
            t[i].fillna(t[i].mode()[0], inplace=True)  # le înlocuim cu moda pentru date categorice

# Aplicăm înlocuirea valorilor lipsă
nan_replace(df_mortalitate)
nan_replace(df_coduri)

# Selectăm toate coloanele cu date, exceptând prima coloană
set_date = list(df_mortalitate.columns)[1:]
x = df_mortalitate[set_date].values

# Standardizăm datele (media 0, deviație standard 1)
scaler = StandardScaler()
date_standardizate = scaler.fit_transform(df_mortalitate)

# Aplicăm ACP
modelACP = PCA()
C = modelACP.fit_transform(date_standardizate)  # Calculăm componentele principale

# Calculăm varianța explicată de fiecare componentă
varianta = modelACP.explained_variance_

# Calculăm scorurile (coordonatele observațiilor în spațiul componentelor)
scoruri = C / np.sqrt(varianta)
etichetari = ['C'+str(i+1) for i in range(len(varianta))]  # Creăm etichete pentru componente
df_scoruri = pd.DataFrame(data=scoruri, index=df_mortalitate.index, columns=etichetari)

# Creăm plot-ul scorurilor pentru primele două componente
plt.figure("Plot scoruri in cele 2 axe principale", figsize=(9,9))
plt.subplot(1,1,1)
plt.xlabel('C1')
plt.ylabel('C2')
plt.title('Plot scoruri in cele doua axe principale')
plt.scatter(df_scoruri['C1'], df_scoruri['C2'], color='r')
for index, (x,y) in df_scoruri[['C1','C2']].iterrows():
    plt.text(x,y,index)

# Calculăm corelațiile dintre variabilele inițiale și componentele principale
r_x_c = np.corrcoef(date_standardizate, C, rowvar=False)[:len(varianta),len(varianta):]
eticheta_corel = ['C'+str(i+1) for i in range(len(varianta))]
df_corelatii = pd.DataFrame(data=r_x_c, index=df_mortalitate.columns, columns=[eticheta_corel])

# Creăm corelograma (heatmap pentru corelații)
plt.figure('Corelograma', figsize=(9,9))
plt.subplot(1,1,1)
sb.heatmap(data=r_x_c, vmin=-1, vmax=1, cmap='bwr', annot=True)
plt.title("Corelograma factoriala")

# Calculăm contribuțiile (cât contribuie fiecare observație la varianta componentei)
C_patrat = C * C
contributii = C_patrat / np.sum(C_patrat, axis=0)
df_contributii = pd.DataFrame(data=contributii, index=df_mortalitate.index, columns=eticheta_corel)

# Calculăm cosinusurile (calitatea reprezentării observațiilor)
cosinusuri = np.transpose(C_patrat.T / np.sum(C_patrat, axis=1))
df_cosinusuri = pd.DataFrame(data=cosinusuri, index=df_mortalitate.index, columns=eticheta_corel)

# Calculăm comunalitățile (proporția din varianța variabilelor explicată de componente)
comunalitati = np.cumsum(r_x_c * r_x_c, axis=1)
df_comunalitati = pd.DataFrame(data=comunalitati, index=df_mortalitate.columns, columns=eticheta_corel)

# Creăm corelograma comunalităților
plt.figure("Corelegrama comunalitati", figsize=(9,9))
plt.subplot(1,1,1)
sb.heatmap(data=df_comunalitati, vmin=-1, vmax=1, cmap='bwr', annot=True)
plt.title("Corelograma comunalitati")
plt.show()