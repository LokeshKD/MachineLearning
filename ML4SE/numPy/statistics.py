import numpy as np


# Analysis

arr = np.array([[0, 72, 3],
                [1, 3, -60],
                [-3, -2, 4]])
print(arr.min())
print(arr.max())

print(repr(arr.min(axis=0)))
print(repr(arr.max(axis=-1)))


# Statistics

arr = np.array([[0, 72, 3],
                [1, 3, -60],
                [-3, -2, 4]])
print(np.mean(arr))
print(np.var(arr))
print(np.median(arr))
print(repr(np.median(arr, axis=-1)))


# Summation

arr = np.array([[0, 72, 3],
                [1, 3, -60],
                [-3, -2, 4]])
print(np.sum(arr))
print(repr(np.sum(arr, axis=0)))
print(repr(np.sum(arr, axis=1)))

arr = np.array([[0, 72, 3],
                [1, 3, -60],
                [-3, -2, 4]])
print(repr(np.cumsum(arr)))
print(repr(np.cumsum(arr, axis=0)))
print(repr(np.cumsum(arr, axis=1)))


# Concatenation

arr1 = np.array([[0, 72, 3],
                 [1, 3, -60],
                 [-3, -2, 4]])
arr2 = np.array([[-15, 6, 1],
                 [8, 9, -4],
                 [5, -21, 18]])
print(repr(np.concatenate([arr1, arr2])))
print(repr(np.concatenate([arr1, arr2], axis=1)))
print(repr(np.concatenate([arr2, arr1], axis=1)))


## Saving Data

arr = np.array([1, 2, 3])
# Saves to 'arr.npy'
np.save('arr.npy', arr)
# Also saves to 'arr.npy'
np.save('arr', arr) # Same as above extension is automatic.

## Loading
arr = np.array([1, 2, 3])
np.save('arr.npy', arr)
load_arr = np.load('arr.npy')
print(repr(load_arr))

# Will result in FileNotFoundError
load_arr = np.load('arr') 


