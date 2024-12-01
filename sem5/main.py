import pandas as pd
import numpy as np

df = pd.read_csv("data/Ethnicity.csv")
loc = pd.read_csv("data/Coduri_Localitati.csv")
regions = pd.read_csv("data/Coduri_Regiuni.csv")
jud = pd.read_csv("data/Coduri_Judete.csv")

# merge everything into a big dataframe
df_merged = df.merge(loc, how="left", on="Code")
df_merged = df_merged.merge(jud, how="left", left_on="County", right_on="IndicativJudet")
df_merged = df_merged.merge(regions, how="left", on="Regiune")

ethnicity_cols = df_merged.loc[:, 'Romanian':'Another'].columns

# Calculate populations by ethnicity for each level
counties_pop = df_merged.groupby("NumeJudet")[ethnicity_cols].sum()
regions_pop = df_merged.groupby("Regiune")[ethnicity_cols].sum()
macroregions_pop = df_merged.groupby("MacroRegiune")[ethnicity_cols].sum()

# Calculate totals
counties_total = counties_pop.sum(axis=1)
regions_total = regions_pop.sum(axis=1)
macroregions_total = macroregions_pop.sum(axis=1)

# Calculate percentages
counties_percentage = counties_pop.div(counties_total, axis=0) * 100
counties_percentage = counties_percentage.round(2)


# Calculate Dissimilarity Index for each ethnicity at county level
def calculate_dissimilarity_index(df, ethnicity_col):
    # x_i is the population of the specific ethnicity in county i
    x_i = df[ethnicity_col]
    # r_i is the rest of population in county i
    r_i = df.sum(axis=1) - x_i

    # T_x is total population of the ethnicity
    T_x = x_i.sum()
    # T_r is total of rest of population
    T_r = r_i.sum()

    # Calculate D index
    D = 0.5 * np.sum(np.abs(x_i / T_x - r_i / T_r))
    return D


# Calculate Shannon-Weaver Index for each county
def calculate_shannon_weaver(row):
    # Calculate proportions (p_i)
    proportions = row / row.sum()
    # Remove zero proportions to avoid log(0)
    proportions = proportions[proportions > 0]
    # Calculate H index
    H = -np.sum(proportions * np.log2(proportions))
    return H


dissimilarity_indices = {}
for ethnicity in ethnicity_cols:
    D = calculate_dissimilarity_index(counties_pop, ethnicity)
    dissimilarity_indices[ethnicity] = round(D, 4)

# Calculate Shannon-Weaver index for each county
shannon_weaver_indices = counties_pop.apply(calculate_shannon_weaver, axis=1).round(4)

# Create DataFrame with results
dissimilarity_df = pd.DataFrame.from_dict(dissimilarity_indices, orient='index', columns=['Dissimilarity Index'])
shannon_weaver_df = pd.DataFrame(shannon_weaver_indices, columns=['Shannon-Weaver Index'])

print("\nDissimilarity Indices by Ethnicity:")
print(dissimilarity_df)
print("\nShannon-Weaver Indices by County:")
print(shannon_weaver_df)

# Optional: Save results to CSV
dissimilarity_df.to_csv('dissimilarity_indices.csv')
shannon_weaver_df.to_csv('shannon_weaver_indices.csv')