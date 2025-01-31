import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

ind_df = pd.read_csv("./data/Industrie.csv", index_col=0)
pop_df = pd.read_csv("./data/PopulatieLocalitati.csv", index_col=0)

# Requirement 1 -------------------

merged_df = ind_df.merge(right=pop_df, right_index=True, left_index=True)

industries_list = ind_df.columns[1:].tolist()


def perCapita(row, industries, pop):
    result = row[industries] / row[pop]
    return pd.Series(
        [row["Localitate_x"]] + result.tolist(), index=["Localitate"] + industries
    )


t1_df = merged_df[["Localitate_x", "Populatie"] + industries_list].apply(
    perCapita, axis=1, industries=industries_list, pop="Populatie"
)
t1_df.to_csv("./dataOUT/Cerinta_1.csv")

# . Să se calculeze și să se salveze în fișierul Cerinta_2.csv activitatea industrială dominantă (cu cifra de afaceri cea
# mai mare) la nivel de județ. Pentru fiecare județ se va afișa indicativul de județ, denumirea activității dominante și
# cifra de afaceri corespunzătoare. (1 punct)

# Requirement 2 -------------------
t2_df = merged_df[industries_list + ["Judet"]].groupby(by="Judet").sum()


def maxOutput(row):
    max_industry = row.idxmax()
    return pd.Series(
        [max_industry, row[max_industry]], index=["Industrie", "Cifra de afaceri"]
    )


t3_df = t2_df[industries_list].apply(func=maxOutput, axis=1)
t3_df.to_csv("./dataOUT/Cerinta_2.csv")

# Requirement 3 -------------------


def clean_data(df):
    assert isinstance(df, pd.DataFrame)

    if df.isna().any().any():
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
    return df


table = pd.read_csv("data/DataSet_34.csv", index_col=0)
table = clean_data(table)

X = StandardScaler().fit_transform(table)

# canonical correlation analysis

cca = CCA(n_components=2)
X_c, Y_c = cca.transform(X, X)
