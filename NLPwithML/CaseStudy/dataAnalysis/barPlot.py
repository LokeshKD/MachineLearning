
## Bar Plotting the Categorical Features

import matplotlib.pyplot as plt

plot_df = final_dataset[['Weekly_Sales', 'Type']]
plot_df = plot_df.groupby('Type').mean()
plot_df.plot.bar()
plt.title('Store Type vs. Weekly Sales')
plt.xlabel('Type')
plt.ylabel('Avg Weekly Sales (Dollars)')
plt.show()
