import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import seaborn as sns


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


# Citirea
df_alcohol = pd.read_csv("data/alcohol.csv", index_col=0)

# Curatare
df_alcohol = clean_data(df_alcohol)

df_alcohol.to_csv("data/results/CleanData.csv")

# Standardizare
numeric_cols = list(df_alcohol.columns[1:])

df_alcohol_numeric = df_alcohol[numeric_cols]


scaler = StandardScaler()
data_standardizat = scaler.fit_transform(df_alcohol_numeric)


df_date_standardizate = pd.DataFrame(data_standardizat, df_alcohol.index, numeric_cols)
df_date_standardizate.to_csv("data/results/standard.csv")

# Dendograma


def create_dendogram(df_data_standardizate):
    methods = ["single", "average", "ward", "complete"]
    linkage_matrices = {}

    for i, method in enumerate(methods, 1):
        plt.figure("Ierarhizarea de clusteri")

        # Ierarhizarea de clusteri
        linkage_matrix = linkage(df_data_standardizate, method=method)
        linkage_matrices[method] = linkage_matrix
        dendrogram(linkage_matrix, truncate_mode="lastp", p=10)
        plt.title("Denograma de clusteri")
        plt.xlabel("Sample data")
        plt.ylabel("Distanta")
        plt.show()
    return linkage_matrices


linkage_matrices = create_dendogram(data_standardizat)
# Determinam numarul optim de clustere pentru analiza ward
linkage_matrix_ward = linkage_matrices["ward"]
distance = linkage_matrix_ward[:, 2]
diference = np.diff(distance, 2)
punctul_elb = np.argmax(diference) + 1
print("Numarul optim de clustere este: ", punctul_elb)

clusters = fcluster(linkage_matrix_ward, t=punctul_elb, criterion="maxclust")

# Partitionarea datelor
df_alcohol["Clusters"] = clusters
print("Datele cu clusterele atribuite:")
print(df_alcohol)
df_alcohol.to_csv("data/results/PartionData.csv")


silhouetteAvg = silhouette_score(df_date_standardizate, clusters)
print("Scorul silhouette: ", silhouetteAvg)
# Scorul Silhouette foarte mic, cum ar fi 0.0139, indică o problemă de separare a clusterelor. Practic, clusterele identificate nu sunt bine definite sau există o suprapunere semnificativă între ele


# 8. Vizualizarea clusterelor în spațiul PCA
modelPCA = PCA()
C = modelPCA.fit_transform(df_date_standardizate)

# Varianta
variance = modelPCA.explained_variance_ratio_
print("Variance: ", variance)

etichetePCA = ["C" + str(i + 1) for i in range(len(numeric_cols))]
df_pca = pd.DataFrame(data=C, index=df_alcohol.index, columns=etichetePCA)
print("Dataframe PCA:")
print(df_pca)

plt.figure("Scatter pentru PCA")
plt.scatter(df_pca["C1"], df_pca["C2"])
plt.title("Scatter")
plt.xlabel("C1")
plt.ylabel("C2")
plt.show()

matice_corelatii = np.corrcoef(df_date_standardizate.T, C.T)[
    : len(numeric_cols), len(numeric_cols) :
]
df_corelatii = pd.DataFrame(
    data=matice_corelatii, index=df_alcohol_numeric.columns, columns=etichetePCA
)
print("Maticea de corelatie:")
print(matice_corelatii)

plt.figure("Heatmap corelatii")
sns.heatmap(df_corelatii, color="y")
plt.title("Heatmap matrice corelatii")
plt.show()

comunalitati = np.cumsum(matice_corelatii**2, axis=1)
df_columalitati = pd.DataFrame(
    data=comunalitati, index=df_alcohol_numeric.columns, columns=etichetePCA
)

plt.figure("Heatmap comunalitati")
sns.heatmap(df_columalitati, color="y")
plt.title("Heatmap matrice comunalitati")
plt.show()
