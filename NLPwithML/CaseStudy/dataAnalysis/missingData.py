
## Merging data
general_features = features_df.columns

print(general_features)
print('General Features: {}\n'.format(general_features.tolist()))

store_features = stores_df.columns
print('Store Features: {}'.format(store_features.tolist()))

### O/P
'''
Index(['Store', 'Date', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2',
       'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment',
       'IsHoliday'],
      dtype='object')
General Features: ['Store', 'Date', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'IsHoliday']

Store Features: ['Store', 'Type', 'Size']
'''

## Merge features
merged_features = features_df.merge(stores_df, on='Store')

print(merged_features)
### O/P
'''
      Store        Date  Temperature  ...  IsHoliday  Type    Size
0         1  2010-02-05        42.31  ...      False     A  151315
1         1  2010-02-12        38.51  ...       True     A  151315
2         1  2010-02-19        39.93  ...      False     A  151315
3         1  2010-02-26        46.63  ...      False     A  151315
4         1  2010-03-05        46.50  ...      False     A  151315
5         1  2010-03-12        57.79  ...      False     A  151315
6         1  2010-03-19        54.58  ...      False     A  151315
...     ...         ...          ...  ...        ...   ...     ...
8183     45  2013-06-14        70.01  ...      False     B  118221
8184     45  2013-06-21        70.13  ...      False     B  118221
8185     45  2013-06-28        76.05  ...      False     B  118221
8186     45  2013-07-05        77.50  ...      False     B  118221
8187     45  2013-07-12        79.37  ...      False     B  118221
8188     45  2013-07-19        82.84  ...      False     B  118221
8189     45  2013-07-26        76.06  ...      False     B  118221

[8190 rows x 14 columns]

'''

## Missing data
na_values = pd.isna(merged_features) # Boolean DataFrame

na_features = na_values.any() # Boolean Series

print(na_features)

### O/P ( True means column contains missing values)
'''
Store           False
Date            False
Temperature     False
Fuel_Price      False
MarkDown1        True
MarkDown2        True
MarkDown3        True
MarkDown4        True
MarkDown5        True
CPI              True
Unemployment     True
IsHoliday       False
Type            False
Size            False
dtype: bool
'''

