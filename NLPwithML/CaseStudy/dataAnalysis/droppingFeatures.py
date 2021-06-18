
## Counting Missing Values

print(len(na_values))
print(sum(na_values['MarkDown1']))
print(sum(na_values['CPI']))

### O/P
'''
8190
4158
585
'''


### Dropping Unusable Features

markdowns = [
    'MarkDown1',
    'MarkDown2',
    'MarkDown3',
    'MarkDown4',
    'MarkDown5'
]

merged_features = merged_features.drop(columns=markdowns)
print(merged_features.columns.tolist())

### O/P
'''
['Store', 'Date', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday', 'Type', 'Size']
'''

