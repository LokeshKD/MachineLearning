
## Checking for categorical data

print(final_dataset['Type'].unique())
print(final_dataset['Dept'].unique())

### O/P
'''
['A' 'B' 'C']
[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 16 17 18 19 20 21 22 23 24 25
 26 27 28 29 30 31 32 33 34 35 36 37 38 40 41 42 44 45 46 47 48 49 51 52
 54 55 56 58 59 60 67 71 72 74 79 80 81 82 83 85 87 90 91 92 93 94 95 97
 98 78 96 99 77 39 50 43 65]
'''

# Update IsHoliday values
# CODE HERE
final_dataset['IsHoliday'] = final_dataset['IsHoliday'].astype(int)

