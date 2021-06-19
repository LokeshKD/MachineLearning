
## Plotting data.

import matplotlib.pyplot as plt

plot_df = final_dataset[['Weekly_Sales', 'Temperature']]
rounded_temp = plot_df['Temperature'].round(0)  # nearest integer
plot_df = plot_df.groupby(rounded_temp).mean()
plot_df.plot.scatter(x='Temperature', y='Weekly_Sales')
plt.show()

## Professional Plotting.
plot_df = final_dataset[['Weekly_Sales', 'Temperature']]
rounded_temp = plot_df['Temperature'].round(0)  # nearest integer
plot_df = plot_df.groupby(rounded_temp).mean()
plot_df.plot.scatter(x='Temperature', y='Weekly_Sales')
plt.title('Temperature vs. Weekly Sales')
plt.xlabel('Temperature (Fahrenheit)')
plt.ylabel('Avg Weekly Sales (Dollars)')
plt.show()

