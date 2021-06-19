
## Finding missing values

import numpy as np  # NumPy library

na_cpi_int = na_values['CPI'].astype(int)
na_indexes_cpi = na_cpi_int.to_numpy().nonzero()[0]
na_une_int = na_values['Unemployment'].astype(int)
na_indexes_une = na_une_int.to_numpy().nonzero()[0]

print(np.array_equal(na_indexes_cpi, na_indexes_une))

na_indexes = na_indexes_cpi
na_rows = merged_features.iloc[na_indexes]

print(na_rows['Date'].unique())  # missing value weeks
print(merged_features['Date'].unique()[-13:])  # final 13 weeks
print(na_rows.groupby('Store').count()['Date'].unique())

print(na_indexes[0])  # first missing value row index
print()
first_missing_row = merged_features.iloc[169]
print(first_missing_row[['Date','CPI','Unemployment']])
print()
final_val_row = merged_features.iloc[168]
print(final_val_row[['Date','CPI','Unemployment']])
print()
cpi_final_val = merged_features.at[168, 'CPI']
une_final_val = merged_features.at[168, 'Unemployment']
merged_features.at[169, 'CPI'] = cpi_final_val
merged_features.at[169, 'Unemployment'] = une_final_val
new_row = merged_features.iloc[169]
print(new_row[['Date','CPI','Unemployment']])
print()

### O/P
'''
169

Date            2013-05-03
CPI                    NaN
Unemployment           NaN
Name: 169, dtype: object

Date            2013-04-26
CPI                 225.17
Unemployment         6.314
Name: 168, dtype: object

Date            2013-05-03
CPI                 225.17
Unemployment         6.314
Name: 169, dtype: object
'''

### Time to Code

import pandas as pd

# Fill in missing data
def impute_data(merged_features, na_indexes_cpi, na_indexes_une):
    # CODE HERE
    for i in na_indexes_cpi:
        merged_features.at[i,'CPI'] = merged_features.at[i-1, 'CPI']
    for i in na_indexes_une:
        merged_features.at[i, 'Unemployment'] = merged_features.at[i - 1, 'Unemployment']

