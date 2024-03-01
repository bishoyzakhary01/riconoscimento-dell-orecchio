import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#dataset = np.load('images.npy',allow_pickle=True)
dataset = np.load('/Users/bishoyzakhary/Downloads/Ape/Images2',allow_pickle=True)

print(f'Shape:{dataset.shape}')

for img in dataset:
    plt.imshow(img)
    plt.show()