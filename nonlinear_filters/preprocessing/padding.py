import numpy as np


def add_padding(image, k):
    sidepadded = np.concatenate([np.flip(image[:,:k], 1), image, np.flip(image[:,image.shape[1]-k:], 1)], axis = 1)
    return np.concatenate([np.flip(sidepadded[:k,:], 0), sidepadded, np.flip(sidepadded[sidepadded.shape[0]-k:,:], 0)], axis = 0)