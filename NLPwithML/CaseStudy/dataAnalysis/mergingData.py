
## Merging the main trainin dataset with its corresponding feature dataset.

train_df = pd.read_csv('weekly_sales.csv')
print(train_df.columns.tolist())

# Merged and imputed stores + features
print(merged_features.columns.tolist())

### O/P
'''
['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday']
['Store', 'Date', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday', 'Type', 'Size']
'''

## 
features = ['Store', 'Date', 'IsHoliday']
final_dataset = train_df.merge(merged_features, on=features)
final_dataset = final_dataset.drop(columns=['Date'])

print(final_dataset.columns.tolist())

### O/P
'''
['Store', 'Dept', 'Weekly_Sales', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Type', 'Size']
'''
