import matplotlib.pyplot as plt

# predefined df
print('{}\n'.format(df))

df.plot()
plt.show()
plt.savefig('df.png')  # save to PNG file

## Labling
# predefined df
print('{}\n'.format(df))

df.plot()
plt.title('HR vs. Year')
plt.xlabel('Year')
plt.ylabel('HR Count')
plt.show()


## Other Plots
# predefined df
print('{}\n'.format(df))

df.plot(kind='hist')
df.plot(kind='box')
plt.show()



