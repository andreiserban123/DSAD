import pandas as pd


df_mortalitate = pd.read_csv("data/Mortalitate.csv", index_col=0)
df_cod = pd.read_csv("data/CoduriTariExtins.csv", index_col=0)


print(df_mortalitate)
print(df_cod)


def clean_data(df):
    assert isinstance(df, pd.DataFrame)
    if df.isna().any().any():
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean()).round(2)
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
    return df


df_mortalitate = clean_data(df_mortalitate)
df_cod = clean_data(df_cod)


df_merge = pd.merge(
    df_mortalitate, df_cod, how="inner", left_index=True, right_index=True
)

df_merge.to_csv("data/AnalyzeDF.csv")
