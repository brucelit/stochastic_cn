import numpy as np

matrix = np.array([
    [1, -1, 0, 0, 0, 0, 0],
    [0, 1,-1, 0, 0, 0, 0],
    [0, 0, 0.75, -0.45, -0.3, 0, 0],
    [0, 0, 0, 1, 0, -1, 0],
    [0, 0, 0, 0, 1, -1, 0],
    [0, 0, 0, 0, 0, 1, -1],
    [0, 0, 0, 0, 0, 0, 1]
], np.float16)

x = np.array([1,1,1.3333333,0.6,0.4,1,1])

print("get the result of matrix mult:",np.dot(matrix,x))