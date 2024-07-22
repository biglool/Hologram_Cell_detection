
import numpy as np
import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches
import time
import keras
from holo_cells.src.training.CustomsLoses import jaccard_loss,specificity, pixel_accuracy,dice_coef_loss,iou,focal_loss

class Detector():
  def __init__(self,modelo,original_shape_cut=(1024,1536),model_shape=(256,384),step_size=(128,128), cellCluster_size=80, min_confidence=0.5):
      np.seterr(divide='ignore', invalid='ignore')
      self.modelo=modelo
      self.reconstructed_model = keras.models.load_model(modelo,safe_mode=False)
      self.reconstructed_model.predict(np.array([np.zeros(model_shape)]),verbose=0)
      self.original_shape_cut=original_shape_cut
      self.step_size= step_size

      self.model_shape=model_shape
      self.resize_factor=original_shape_cut[0]/model_shape[0]
      self.min_cellCluster_size=cellCluster_size
      self.min_confidence = min_confidence

  def normalize_images(self,array):
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_array = 2 * (array - min_val) / (max_val - min_val) - 1
    return scaled_array

  def current_milli_time(self):
      return round(time.time() * 1000)

  def get_images_predict_and_print(self,carpeta_reals, print_times=False):
      for f in os.listdir(carpeta_reals):
        im = Image.open(carpeta_reals + f)
        if print_times:
          initialtime=self.current_milli_time()

        sub_imgs,cuts, posicions=self.get_subsegments(im)
        if print_times:
          sub_time=self.current_milli_time()
          print(" total tiempo subdiv:" +str(sub_time-initialtime))

        predicts=self.predict(sub_imgs)

        if print_times:
          print(" total tiempo pred:" +str(self.current_milli_time()-sub_time))
          sub_time=self.current_milli_time()

        full_pred=self.reconstruct_pred(predicts,cuts, posicions)
        if print_times:
          print(" total tiempo recons:" +str(self.current_milli_time()-sub_time))
          sub_time=self.current_milli_time()

        boxes=self.get_boxes(full_pred)
        if print_times:
          print(" total tiempo cluster:" +str(self.current_milli_time()-sub_time))
          sub_time=self.current_milli_time()

        if print_times:
          print(" total tiempo:" +str(self.current_milli_time()-initialtime))
        self.print_result(np.array(im),boxes)

        return boxes, full_pred

  def get_subsegments(self, im):
      images = []
      size_cut_y = self.original_shape_cut[0]
      size_cut_x = self.original_shape_cut[1]
      step_size_y =self.step_size[0]
      step_size_x = self.step_size[1] # int(size_cut_x * self.step)
      rez_fac = self.resize_factor

      # Convert image to NumPy array
      im_np = np.array(im)
      #median_value = np.median(im_np) si queremos recortar boders
      # Cut subsegments of self.original_size_cut
      x_cuts=im_np.shape[0]//step_size_x+1
      y_cuts=im_np.shape[0]//step_size_y+1
      x_size=step_size_x*x_cuts
      y_size=step_size_y*y_cuts

      cuts = [x_cuts, y_cuts]
      positions = []

      for i in range(0, x_size, step_size_x):
          for j in range(0, y_size, step_size_y):
              if i  < x_size and j  < y_size:
                  positions.append([i / rez_fac, j / rez_fac])
                  subsegment=np.zeros((size_cut_y,size_cut_x))
                  cut=im_np[j:j + size_cut_y, i:i + size_cut_x]
                  subsegment[0:cut.shape[0], 0:cut.shape[1]] = cut
                  resized_sub = cv2.resize(subsegment, (self.model_shape[1], self.model_shape[0]))
                  images.append(self.normalize_images(resized_sub))

      return images, cuts, positions

  def predict(self,images):
      predicts=self.reconstructed_model.predict(np.array(images),verbose=0)
      return predicts

  def reconstruct_pred(self,predicts,cuts,posicions):
      size_cut_y=self.model_shape[0]
      size_cut_x=self.model_shape[1]
      step_size_y=int(self.step_size[0]/(self.original_shape_cut[0]/size_cut_y))
      step_size_x=int(self.step_size[1]/(self.original_shape_cut[1]/size_cut_x))

      reconstructed_prediction=np.zeros(((cuts[1]+1)*step_size_y, (cuts[0]+1)*step_size_x))
      for k,i in enumerate(posicions):
        x=int(i[0])
        y=int(i[1])

        reconstructed_prediction[y:y+size_cut_y,x:x+size_cut_x]+=predicts[k].reshape(size_cut_y,size_cut_x)

      return reconstructed_prediction

  def get_boxes(self,reconstructed_prediction):
    confidence = self.min_confidence
    positions = np.argwhere(reconstructed_prediction >confidence) #get x, y of pixels detected with conf >0.5
    rez=self.resize_factor
    boxes=[]
    if len(positions) > 0: #at least one pixel detected
        db = DBSCAN(eps=1.1, min_samples=5).fit(positions)
        labels = db.labels_
        info=np.unique(labels, return_counts=True)
        boxes=[]

        #plot de piwel seleccion
        #fig, ax =  plt.subplots(1, figsize=(10, 10))
        #pixel_x=[]
        #pixel_y=[]
        #for x,y in positions:
         # pixel_x.append(x)
         # pixel_y.append(y)
        #ax.scatter(pixel_x, pixel_y, c= labels)
        #plt.show()

        for cluster, cuentas in  zip(info[0], info[1]):
          if cluster >=0 and cuentas>self.min_cellCluster_size:
            mascara=np.argwhere(labels==cluster)
            pixels=positions[mascara].reshape(-1,2)
            x_values = pixels[:, 1]
            y_values = pixels[:, 0]
            boxes.append([np.min(x_values)*rez,np.min(y_values)*rez,np.max(x_values)*rez,np.max(y_values)*rez])


    return boxes

  def print_result(self,img,boxes):
      fig, ax =  plt.subplots(1, figsize=(10, 10))
      print(img.shape)
      ax.imshow(img)
      for box in boxes:
        x, y, x2, y2 = box
        rect = patches.Rectangle((x, y), x2-x, y2-y, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

      plt.show()
