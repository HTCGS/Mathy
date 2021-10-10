import numpy as np
import matplotlib.pyplot as plt


def merge_sort(arr):
    result = np.copy(arr)
    if len(arr) > 1:
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]
        L = merge_sort(L)
        R = merge_sort(R)

        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                result[k] = L[i]
                i += 1
            else:
                result[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            result[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            result[k] = R[j]
            j += 1
            k += 1
    return result


arr = np.array([2, 1, 4, 3])
arr = merge_sort(arr)
print(arr)
