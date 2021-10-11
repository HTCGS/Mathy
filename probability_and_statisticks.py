import calculus
import numpy as np
import matplotlib.pyplot as plt


# nums = np.array([1, 1, 2, 2, 4, 7, 7, 7, 8, 9])
# mean = nums.mean()
# print(mean)
# median = np.median(nums)
# print(median)
# std = np.std(nums)
# print(std)

coins = np.array([[0, 0.25], [1, 0.5], [2, 0.25]])
mean = coins.mean()
print(mean)
var = np.var(coins)
print(var)
std = np.std(coins)
print(std)

# plt.figure()
# plt.bar(["0", "1", "2"], coins)
# plt.show()

#! comment
