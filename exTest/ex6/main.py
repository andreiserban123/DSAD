import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


#   1.	Citirea fișierelor CSV și analiza inițială a datelor
# 	    •	Citește fișierele vanzari.csv și produse.csv.
# 	    •	Afișează primele 5 rânduri din fiecare fișier.
# 	    •	Verifică dacă există valori lipsă și afișează numărul acestora pentru fiecare coloană.
# 	2.	Îmbinarea datelor (join)
# 	    •	Realizează un merge între cele două DataFrame-uri folosind coloana comună ID_Produs.
# 	    •	Afișează structura tabelului rezultat după merge si salveaza-l întru-un fișier csv
# 	3.	Curățarea datelor
# 	    •	Completează valorile lipsă:
# 	    •	Pentru coloanele numerice, completează cu media lor.
# 	    •	Pentru coloanele categorice, completează cu cel mai frecvent element (modulul).
# 	4.	Standardizarea datelor
# 	    •	Identifică toate coloanele numerice din DataFrame-ul final.
# 	    •	Standardizează fiecare coloană numerică astfel încât să aibă media 0 și abaterea standard 1.
# 	5.	Analiza corelației
# 	    •	Calculează matricea de corelație între variabilele numerice.
# 	    •	Creează un heatmap pentru a vizualiza corelațiile.
# 	6.	Vizualizarea distribuției
# 	    •	Creează un grafic de distribuție (histogramă) pentru coloana Profit.

def standardizeData(df, numericCols):
    assert isinstance(df, pd.DataFrame)
    for col in numericCols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
    return df

# 1. Citire date
df_prod = pd.read_csv('data/produse.csv', index_col=0)
df_vanz = pd.read_csv('data/vanzari.csv', index_col=0)

# 2. Merge
df_merged = df_vanz.merge(df_prod, left_index=True, right_index=True)
df_merged.to_csv('data/output/merged.csv')

# 3. Curățare date
def nan_replace(t):
    assert isinstance(t, pd.DataFrame)
    for i in t.columns:
        if t[i].isna().any():  # dacă există valori lipsă
            if np.issubdtype(t[i].dtype, np.number):  # verificăm dacă coloana e numerică
                t[i] = t[i].fillna(t[i].mean())
            else:  # pentru coloane non-numerice
                t[i] = t[i].fillna(t[i].mode()[0])
    return t

df_merged = nan_replace(df_merged)
df_merged.to_csv('data/output/merged_clean.csv')

# 4. Standardizare
# Selectăm doar coloanele numerice
numeric_columns = df_merged.select_dtypes(include=[np.number]).columns
standardizedDatabase = standardizeData(df_merged.copy(True), numeric_columns)
standardizedDatabase.to_csv('data/output/standardized.csv')

# 5. Analiza corelației
matrice_corelatie = df_merged[numeric_columns].corr()
plt.figure(figsize=(12, 8))  # Setăm dimensiunea figurii pentru o vizualizare mai bună
sns.heatmap(matrice_corelatie,
            annot=True,      # Afișează valorile numerice
            cmap='coolwarm', # Schema de culori: roșu pentru corelații pozitive, albastru pentru negative
            center=0,        # Centrul scalei de culori la 0
            vmin=-1,         # Valoarea minimă pentru scală
            vmax=1)          # Valoarea maximă pentru scală

plt.title('Matricea de Corelație între Variabilele Numerice')
plt.tight_layout()  # Ajustează automat spațierea
plt.show()

# 6. Vizualizarea distribuției
# Set the visual style for better aesthetics
sns.histplot(standardizedDatabase["Profit"], kde = True, bins = 10, color="r")
plt.title("Distribuția Profitului")
plt.xlabel("Profit (standardizat)")
plt.ylabel("Frecvență")

# •    Profiturile sunt distribuite în jurul mediei standardizate(0), cu câteva valori extreme spre stânga și dreapta.
# •    Curba KDE confirmă că cele mai multe valori se află aproape de media generală.
# •    Distribuția arată o ușoară variație, dar nu este uniformă sau perfect normală.
plt.show()


# profit nestandardizat

sns.histplot(df_merged["Profit"], kde = True, bins = 10, color="r")
plt.title("Distribuția Profitului")
plt.xlabel("Profit")
plt.ylabel("Frecvență")
plt.show()