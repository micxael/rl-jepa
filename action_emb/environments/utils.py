import numpy as np


def binary_encoding(number, size):
    binary = np.zeros(size)
    i = -1
    while number > 0:
        binary[i] = number % 2
        number = number // 2
        i -= 1
    return binary


def in_region(pos, region):
    x0, y0 = pos
    x1, y1, x2, y2 = region
    if x0 >= x1 and x0 <= x2 and y0 >= y1 and y0 <= y2:
        return True
    else:
        return False


class Space:
    def __init__(self, low=[0], high=[1], dtype=np.uint8, size=-1):
        if size == -1:
            self.shape = np.shape(low)
        else:
            self.shape = (size, )
        self.low = np.array(low)
        self.high = np.array(high)
        self.dtype = dtype
        self.n = len(self.low)

