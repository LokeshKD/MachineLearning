import pandas as pd

# Quantitative Features
df = pd.DataFrame({
  'T1': [10, 15, 8],
  'T2': [25, 27, 25],
  'T3': [16, 15, 10]})

print('{}\n'.format(df))

print('{}\n'.format(df.sum()))

print('{}\n'.format(df.sum(axis=1)))

print('{}\n'.format(df.mean()))

print('{}\n'.format(df.mean(axis=1)))


# Weighted Features
df = pd.DataFrame({
  'T1': [0.1, 150.],
  'T2': [0.25, 240.],
  'T3': [0.16, 100.]})

print('{}\n'.format(df))

print('{}\n'.format(df.multiply(2)))

df_ms = df.multiply([1000, 1], axis=0)
print('{}\n'.format(df_ms))

df_w = df_ms.multiply([1,0.5,1])
print('{}\n'.format(df_w))
print('{}\n'.format(df_w.sum(axis=1)))


