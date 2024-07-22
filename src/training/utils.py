

import numpy as np
import matplotlib.pyplot as plt
import tqdm as tq
import zipfile


def unzip(origin,destino):
  with zipfile.ZipFile(origin, 'r') as zip_ref:
      zip_ref.extractall(destino)

def showBatch(batch):
   
    for x,y in zip(batch[0],batch[1]):
        print("imagen")
        plt.imshow(x)
        plt.show()
        print("mascara")
        plt.imshow(y)

        plt.show()
        print(np.unique(y,return_counts=True))