
import os
import matplotlib as plt
from PIL import Image
import matplotlib.image as mpimg
import numpy as np

def Visualiza_carpeta_real(detector, mostres):
    for carp in mostres:
        carpeta_reals = carp
        boxes, pred_masc =detector.get_images_predict_and_print(carpeta_reals+"/",print_times=True)

        plt.imshow(pred_masc)
        plt.show()

        #boxes sobre el holograma projectat
        holograma = carpeta_reals+"_hologram/"
        if os.path.exists(holograma):

            for f in os.listdir(holograma):
                if f.endswith(".tiff"):
                    im=Image.open(holograma + f)
                    plt.imshow(np.array(im))
                    plt.show()
                if f.endswith(".png"):

                    im = mpimg.imread(holograma + f)
                    plt.imshow(im)
                    plt.axis('off')
                    plt.show()