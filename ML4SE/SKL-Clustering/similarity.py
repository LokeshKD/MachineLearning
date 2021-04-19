
## Cosine Similarity
# values between -1 and 1
# 1 - Very similar, 0 - no corelation, -1 - divergent

from sklearn.metrics.pairwise import cosine_similarity

data = np.array([
  [ 1.1,  0.3],
  [ 2.1,  0.6],
  [-1.1, -0.4],
  [ 0. , -3.2]])
cos_sims = cosine_similarity(data)
print('{}\n'.format(repr(cos_sims)))


data2 = np.array([
  [ 1.7,  0.4],
  [ 4.2, 1.25],
  [-8.1,  1.2]])
cos_sims = cosine_similarity(data, data2)
print('{}\n'.format(repr(cos_sims)))
