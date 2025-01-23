import pandas as pd


def cleanData(dataDF):
    isinstance(dataDF, pd.DataFrame)
    if dataDF.isna().any().any():
        print("Cleaning data")
        for col in dataDF.columns:
            if dataDF[col].isna().any():
                if pd.api.types.is_numeric_dtype(dataDF[col]):
                    dataDF[col] = dataDF[col].fillna(dataDF[col].mean)
                else:
                    dataDF[col] = dataDF[col].fillna(dataDF[col].mode()[0])

    return dataDF


# 1. Citim datele
df_mortalitate = pd.read_csv("data/Mortalitate.csv", index_col=0)
df_coduri = pd.read_csv("data/CoduriTariExtins.csv", index_col=0)

# 2. Curatam datele
df_mortalitate = cleanData(df_mortalitate)
df_coduri = cleanData(df_coduri)

print(df_mortalitate)
