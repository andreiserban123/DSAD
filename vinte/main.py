import pandas as pd
import numpy as np


ind_df = pd.read_csv("data/Industrie.csv", index_col=0)
pop_df = pd.read_csv("data/PopulatieLocalitati.csv", index_col=0)

# Requirement 1 -------------------

merged_df = ind_df.merge(right=pop_df, right_index=True, left_index=True)

print(merged_df)
