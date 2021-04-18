import pandas as pd

# Read CSV
# data.csv contains baseball data
df = pd.read_csv('data.csv')
# Newline to separate print statements
print('{}\n'.format(df))

df = pd.read_csv('data.csv', index_col=0)
print('{}\n'.format(df))

df = pd.read_csv('data.csv', index_col=1)
print('{}\n'.format(df))

# Read Excel
# data.csv contains baseball data
df = pd.read_excel('data.xlsx')
# Newline to separate print statements
print('{}\n'.format(df))

print('Sheet 1 (0-indexed) DataFrame:')
df = pd.read_excel('data.xlsx', sheet_name=1)
print('{}\n'.format(df))

print('MIL DataFrame:')
df = pd.read_excel('data.xlsx', sheet_name='MIL')
print('{}\n'.format(df))

# Sheets 0 and 1
df_dict = pd.read_excel('data.xlsx', sheet_name=[0, 1])
print('{}\n'.format(df_dict[1]))

# All Sheets
df_dict = pd.read_excel('data.xlsx', sheet_name=None)
print(df_dict.keys())

# Read JSON
# data is the JSON data (as a Python dict)
print('{}\n'.format(data))

df1 = pd.read_json('data.json')
print('{}\n'.format(df1))

df2 = pd.read_json('data.json', orient='index')
print('{}\n'.format(df2))


# Write to JSON
'''
The to_json function also uses the orient keyword argument that was part of pd.read_json. 
Like in pd.read_json, setting orient='index' will set the outer keys of the JSON data to 
the row labels and the inner keys to the column labels.
'''

# Predefined df
print('{}\n'.format(df))

df.to_json('data.json')
df2 = pd.read_json('data.json')
print('{}\n'.format(df2))

df.to_json('data.json', orient='index')
df2 = pd.read_json('data.json')
print('{}\n'.format(df2))
df2 = pd.read_json('data.json', orient='index')
print('{}\n'.format(df2))
