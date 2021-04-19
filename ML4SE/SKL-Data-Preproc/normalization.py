
## Data Normalization
'''

X = [x1,x2,...xm]

l = RMS(xi)
X(L2) = [x1/l,x2/l,...xm/l]
'''

# predefined data
print('{}\n'.format(repr(data)))

from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
transformed = normalizer.fit_transform(data)
print('{}\n'.format(repr(transformed)))



### Imputation

# predefined data
print('{}\n'.format(repr(data)))

from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer() # mean imputation
transformed = imp_mean.fit_transform(data)
print('{}\n'.format(repr(transformed)))


imp_median = SimpleImputer(strategy='median')
transformed = imp_median.fit_transform(data)
print('{}\n'.format(repr(transformed)))

imp_frequent = SimpleImputer(strategy='most_frequent')
transformed = imp_frequent.fit_transform(data)
print('{}\n'.format(repr(transformed)))

imp_constant = SimpleImputer(strategy='constant',
                             fill_value=-1)
transformed = imp_constant.fit_transform(data)
print('{}\n'.format(repr(transformed)))

