import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def cleanData(dataDF):
    assert isinstance(dataDF, pd.DataFrame)
    if dataDF.isna().any().any():
        print("Cleaning data")
        for col in dataDF.columns:
            if dataDF[col].isna().any():
                if pd.api.types.is_numeric_dtype(dataDF[col]):
                    dataDF[col] = dataDF[col].fillna(dataDF[col].mean())
                else:
                    dataDF[col] = dataDF[col].fillna(dataDF[col].mode()[0])

    return dataDF


# 1. Citim datele
df_mortalitate = pd.read_csv("data/Mortalitate.csv", index_col=0)
df_coduri = pd.read_csv("data/CoduriTariExtins.csv", index_col=0)

# 2. Curatam datele
df_mortalitate = cleanData(df_mortalitate)
df_coduri = cleanData(df_coduri)

print("Dataframe-uri curatate")

print("DF mortalitate: ")
print(df_mortalitate)

print("DF Coduri:")
print(df_coduri)

# 3. Facem merge de date
df_combined = df_mortalitate.merge(df_coduri, right_index=True, left_index=True)

print("Dataframe merge-uit:")
print(df_combined)
df_combined.to_csv("data/DataframeForAnalysis.csv")

# 4. Standardizam datele numerice

df_numeric = df_combined.select_dtypes(include=["float64", "int64"])
scaler = StandardScaler()
date_standard = scaler.fit_transform(df_numeric)


# 5. Construim modelul ACP
modelACP = PCA()
C = modelACP.fit_transform(date_standard)

# 6. Varianta componetelor principale
# Aceste valori sunt varianța explicată sau proporția variabilității explicate de fiecare componentă:
# •	0.38098545 → Prima componentă principală (C1) explică 38,1% din variabilitatea totală a datelor.
# •	0.16654193 → A doua componentă principală (C2) explică 16,7% din variabilitatea totală.


variance = modelACP.explained_variance_ratio_
print("Varianta componentelor principale")
print(variance)

# Facem un dataframe cu scorurile
componente = ["C" + str(i + 1) for i in range(len(variance))]

df_scoruri = pd.DataFrame(data=C, index=df_combined.index, columns=componente)
print("Scorurile componentelor principale")
print(df_scoruri)


# 7. Vizualizarea componentelor principale
indicator_names = {
    "C1": "Rate of Change (RS)",
    "C2": "Fertility Rate (FR)",
    "C3": "Infant Mortality (IM)",
    "C4": "Maternal Mortality (MMR)",
    "C5": "Life Expectancy (LE)",
    "C6": "Life Expectancy Male (LEM)",
    "C7": "Life Expectancy Female (LEF)",
}

plt.figure("Principal Components Plot")
plt.scatter(df_scoruri["C1"], df_scoruri["C2"], color="r")
plt.xlabel(indicator_names["C1"])
plt.ylabel(indicator_names["C2"])
plt.title("Principal Components Analysis")
plt.show()

# 8. Corelatii factoriale
# Corelații între variabilele originale și componentele principale
r_x_c = np.corrcoef(date_standard.T, C.T)[
    : len(df_numeric.columns), len(df_numeric.columns) :
]
df_corelatii = pd.DataFrame(data=r_x_c, index=df_numeric.columns, columns=componente)
print("Corelațiile factoriale:")
print(df_corelatii)
# 9. Corelograma factoriala
plt.figure("Corelograma factoriala")
sns.heatmap(df_corelatii)
plt.title("Corelograma factoriala")
plt.show()

# 10. Comunalități
comunalitati = np.cumsum(r_x_c**2, axis=1)
df_comunalitati = pd.DataFrame(
    data=comunalitati, index=df_numeric.columns, columns=componente
)

plt.figure("Corelograma Comunalități")
sns.heatmap(df_comunalitati)
plt.title("Corelograma Comunalități")
plt.show()
