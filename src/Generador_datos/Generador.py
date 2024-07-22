
import numpy as np
import io
import os
from tqdm import tqdm
import pickle
import cv2
import matplotlib as plt
from matplotlib import patches
from scipy.ndimage import zoom
from holo_cells.src.Generador_datos.unxip_auxiliars import propagator_planar_fresnel, ift2_dc,ft2_dc


def genera_guarda_holograma(id,carpeta_base,carpeta,min_prof,max_prof,img_size,margin, mount):


    measured_image_sumada = np.zeros((img_size[1],img_size[0]))
    hologramas=[]
    parametros=[]
    mascaras=[]

    profundidades=[np.random.randint(min_prof, max_prof) for i in range(3)]
    for prof in profundidades:
      num_parts=np.random.randint(3, 6)


      particles=generate_random_cells(num_parts, img_size, margin,prof, 14,11,0.5)

      mount['distance_display_sample']=prof

      mascara= genera_mascara(img_size, particles)
      hologramas.append(generate_hologram(mascara, mount))
      mascaras.append(mascara)
      #parametros.append(particles)
      for part in particles:
        x_p, y_p,z,r_p ,r_in_p = part
        parametros.append([-(x_p/4) +int(img_size[0]/8), -(y_p/4)+int(img_size[1]/8),z,r_p/4,r_in_p])

    measured_image_sumada= np.zeros((img_size[1],img_size[0]))
    for holo in hologramas:
      measured_image_sumada=measured_image_sumada+holo

    mascara_final= np.ones((img_size[1],img_size[0]))
    for masc in mascaras:
      mascara_final=np.minimum(mascara_final,masc)


    mascara_final=cv2.resize(np.where(mascara_final == 1, 0, 1).astype(np.uint8), (int(img_size[0]/4),int(img_size[1]/4)))
    measured_image_sumada=cv2.resize(np.abs(measured_image_sumada), (int(img_size[0]/4),int(img_size[1]/4)))

    if not os.path.exists(carpeta_base+"/mascaras/"+carpeta[0]+"/"): os.makedirs(carpeta_base+"/mascaras/"+carpeta[0]+"/")
    if not os.path.exists(carpeta_base+"/hologramas/"+carpeta[0]+"/"): os.makedirs(carpeta_base+"/hologramas/"+carpeta[0]+"/")
    if not os.path.exists(carpeta_base+"/parametros/"+carpeta[0]+"/"): os.makedirs(carpeta_base+"/parametros/"+carpeta[0]+"/")




def rescala_holos(holo,mascara,particulas, tamanyo):
    #asuminos holos quadrados

    zoom_factor =tamanyo/holo.shape[0]
    res_holo = zoom(holo, zoom_factor)
    res_mascara=zoom(mascara, zoom_factor)
    res_parts=[]
    for part in particulas:
      x_p, y_p,z,r_p ,r_in_p = part
      res_parts.append([x_p*zoom_factor, y_p*zoom_factor, z, r_p*zoom_factor, r_in_p*zoom_factor])
    return res_holo,res_mascara,res_parts

def subdivide( holo, mascara , particles, tamany):



    #[x,y,z,radius,radius_internal]
    #plt.imshow(holo)
    #plt.show()

    #overlay_images(mascara, particles,(mascara.shape[1],mascara.shape[0]))
    # Calculate the number of subdivisions
    subs_alt = holo.shape[0] // tamany
    subs_llarg = holo.shape[1] // tamany

    # Subdivide the array
    holo_parts = []
    mascara_parts=[]
    selected_particles = []

    for i in range(subs_alt):
        for j in range(subs_llarg):
            nom= str(id)+"_"+str(i)+ "_"+ str(j)

            x_start = j * tamany
            x_end = (j + 1) * tamany
            y_start = i * tamany
            y_end = (i + 1) * tamany

            sub_holo = holo[y_start:y_end, x_start:x_end]

            holo_parts.append(sub_holo)
            sub_masc = mascara[y_start:y_end, x_start:x_end]
            mascara_parts.append(sub_masc)

            # Select particles within the boundaries of the current patch
            relevant_particles = []
            for particle in particles:

                x, y, z, radius, radius_internal = particle
                # Adjust x and y relative to the center of the image
                x_rel =  (holo.shape[1] / 2)-x
                y_rel =  (holo.shape[0] / 2)-y

                if x_start <= x_rel < x_end and y_start <= y_rel < y_end:

                    # Calculate relative position within the patch
                    x_patch_rel = x_rel - x_start
                    y_patch_rel = y_rel - y_start
                    #print([x_patch_rel, y_patch_rel, z, radius, radius_internal])
                    relevant_particles.append([x_patch_rel, y_patch_rel, z, radius, radius_internal])

            selected_particles.append(relevant_particles)

            #fig, ax = plt.subplots(figsize=(10, 10))
            #ax.imshow(np.abs(sub_holo))#, cmap='gray'

            # Overlay particles from ground truth with red outlines
            #for particle in relevant_particles:
             #   x_p, y_p,z,r_p ,r_in_p = particle
              #  rect=patches.Rectangle(( x_p -r_p, y_p -r_p), r_p*2, r_p*2, linewidth=1, edgecolor='r', facecolor='none')
              #  ax.add_patch(rect)
           # plt.show()
    return holo_parts,mascara_parts,selected_particles

def  genera_mascara( img_size, particles):
   # Define some data:
  N = img_size[0]
  M = img_size[1]

  # Define a meshgrid:
  XX, YY = np.meshgrid(np.arange(N), np.arange(M))

  particulas =[]
  particulas_interior=[]
  for x,y,z,r,r_in in particles:

    m = (XX-(N//2)+ x)**2 + (YY-(M//2)+ y)**2 > r**2
    m_in = (XX-(N//2)+x)**2 + (YY-(M//2)+y)**2 > r_in**2

    if len(particulas)==0:
      particulas=m
    else:
      particulas=particulas*m

    if len(particulas_interior)==0:
      particulas_interior=m_in
    else:
      particulas_interior=particulas_interior*m_in



  sim_image = (particulas) .astype(float)
  sim_image[~(particulas_interior )] = 0.9
  return sim_image


def generate_hologram( sim_image,mount ):


  # Rename measured distances:
  z = mount.get("distance_display_camera") * 1e-3
  z0 = mount.get("distance_display_sample") * 1e-6
  h = sim_image.shape[0] * mount.get("pitch_camera") * 1e-6

  # Convert the height of the sensor plane to the height in the hologram plane:
  size_y = z0 * h / z

  # Compute the area in the x-axis:
  size_x = size_y * sim_image.shape[1] / sim_image.shape[0]

  propagator = propagator_planar_fresnel(
      sim_image.shape[0],
      sim_image.shape[1],
      mount.get("wavelength") * 1e-9,
      size_x,
      size_y,
      z0
  )

  # Simulate the image by propagating the field:
  measured_image = ift2_dc(ft2_dc(sim_image) * propagator)

  return measured_image


def generate_random_cells(num_cels,img_size,margin,z,max_radius,min_radius, percen_internal):
  # particulas [x, y , radi, radi intern, transparencia]
  cells=[]
  max_x = img_size[0]/2 - margin

  min_x=-max_x
  max_y=img_size[1]/2 -margin
  min_y=-max_y

  for n_cel in range(num_cels):
    #print(n_cel)
    x=np.random.randint(min_x, max_x)
    y=np.random.randint(min_y, max_y)

    radius = max_radius if min_radius==max_radius else np.random.randint(min_radius, max_radius)

    cells.append([x,y,z,radius,radius*percen_internal])

  return np.array(cells)

def add_noise_to_array(array, noise_level):
    """
    Add Gaussian noise to a 2-dimensional array.

    Parameters:
    array (list of lists): The input 2D array.
    noise_level (float): The standard deviation of the Gaussian noise.

    Returns:
    np.array: The modified array with noise added.
    """
    # Convert the input list of lists to a numpy array
    array = np.array(array, dtype=float)

    # Generate Gaussian noise
    noise = np.random.normal(0, noise_level, array.shape)

    # Add the noise to the original array
    noisy_array = array + noise

    return noisy_array

def overlay_images(intensity, particles, image_size):
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(intensity.real)#, cmap='gray'

    # Overlay particles from ground truth with red outlines
    for particle in particles:
        x_p, y_p,z,r_p ,r_in_p = particle
        adjusted_size = r_p # * (100 / (z_p + 1))  # Adjust size based on distance (z_p)
        # Adjust particle positions to image coordinates
        rect=patches.Rectangle(((image_size[0] / 2 - x_p )-r_p,( image_size[1] / 2 - y_p )-r_p), r_p*2, r_p*2, linewidth=1, edgecolor='r', facecolor='none')
        #circle = patches.Circle(((image_size[0] / 2 - x_p ),( image_size[1] / 2 - y_p )), radius=adjusted_size/2 , edgecolor='red', facecolor='none', linewidth=1.5)
        ax.add_patch(rect)

    plt.title('Overlay of Ground Truth on Diffraction Image')
    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.show()


def scale_array( array):
        min_val = np.min(array)
        max_val = np.max(array)
        scaled_array = 2 * (array - min_val) / (max_val - min_val) - 1
        return scaled_array

