
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


def genera_guarda_holograma(id,carpeta_base,carpeta,min_prof,max_prof,min_size, max_size,img_size,margin, mount):


    measured_image_sumada = np.zeros((img_size[1],img_size[0]))
    hologramas=[]
    parametros=[]
    mascaras=[]

    profundidades=[np.random.randint(min_prof, max_prof) for i in range(3)]
    for prof in profundidades:
      num_parts=np.random.randint(3, 6)


      particles=generate_random_cells(num_parts, img_size, margin,prof, max_size,min_size,0.5)

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




def generate_circle_particle(grid_size=40, border_width=5, mid_value=0.5 ):
    # Calculate the radius and center (20.5, 20.5)
    radius = grid_size // 2
    center = (radius - 0.5, radius - 0.5)

    # Create a grid of zeros
    grid = np.zeros((grid_size, grid_size))

    # Populate the grid with 1s for the border and 0.5 for the inside of the circle
    for x in range(grid_size):
        for y in range(grid_size):
            dist_squared = (x - center[0])**2 + (y - center[1])**2
            # Border points (with configurable thickness)
            if (radius - border_width)**2 <= dist_squared <= radius**2:
                grid[x, y] = 1
            # Inside the circle
            elif dist_squared < (radius - border_width)**2:
                grid[x, y] = mid_value

    return grid


import numpy as np
import random
from scipy.spatial import ConvexHull
from skimage.draw import polygon
import math

def generate_random_shape(grid_size=40, num_points=20, border_width=4, radius_variation=0.2):
    # Create an empty grid
    grid = np.zeros((grid_size, grid_size))
    
    # Center of the shape
    center_x, center_y = grid_size // 2, grid_size // 2
    
    # Generate random points around the center within a certain radius
    points = []
    base_radius = grid_size // 3  # Radius for generating points
    for _ in range(num_points):
        angle = random.uniform(0, 2 * np.pi)  # Spread points in all directions
        # Limit the radius variation to make the shape more round
        r = base_radius * (1 + random.uniform(-radius_variation, radius_variation))
        x = center_x + r * np.cos(angle)
        y = center_y + r * np.sin(angle)
        points.append([x, y])
    
    points = np.array(points)
    
    # Get the convex hull of the points to form a closed shape
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # Create a filled polygon from the hull points
    rr, cc = polygon(hull_points[:, 0], hull_points[:, 1], grid.shape)
    grid[rr, cc] = 0.2  # Fill inside the shape with 0.5
    
    # Border: apply a simple dilation or offset to create the border effect
    border_mask = np.zeros_like(grid)
    rr, cc = polygon(hull_points[:, 0], hull_points[:, 1], grid.shape)
    border_mask[rr, cc] = 1
    
    # Apply border width by adding the border as 1 around the shape
    from scipy.ndimage import binary_dilation
    for _ in range(border_width):
        border_mask = binary_dilation(border_mask)
    
    grid[border_mask == 1] = 1
    
    return grid




def generate_ellipse_particle(grid_size=40, r_x=20, r_y=15, border_width=6, mid_value=0.7, angle=45):
    # Center of the ellipse (20.5, 20.5)
    center = (grid_size // 2 - 0.5, grid_size // 2 - 0.5)

    # Convert the rotation angle to radians
    angle_rad = math.radians(angle)

    # Create a grid of zeros
    grid = np.zeros((grid_size, grid_size))

    # Populate the grid with 1s for the border and 0.5 for the inside of the ellipse
    for x in range(grid_size):
        for y in range(grid_size):
            # Translate the grid points to be relative to the center
            x_trans = x - center[0]
            y_trans = y - center[1]

            # Apply rotation to the translated coordinates
            x_rot = x_trans * math.cos(angle_rad) - y_trans * math.sin(angle_rad)
            y_rot = x_trans * math.sin(angle_rad) + y_trans * math.cos(angle_rad)

            # Calculate the distance in the rotated space using the ellipse equation
            dist = (x_rot**2 / r_x**2) + (y_rot**2 / r_y**2)

            # Border points (with configurable thickness)
            if 1 - (border_width / min(r_x, r_y)) <= dist <= 1:
                grid[x, y] = 1
            # Inside the ellipse
            elif dist < 1 - (border_width / min(r_x, r_y)):
                grid[x, y] = mid_value

    return grid


  
def generate_random_shape2(grid_size=40, border_width=6, mid_value=0.5):
    # Create a grid of zeros (transparent background)
    grid = np.zeros((grid_size, grid_size))
    
    # Center of the grid
    center = (grid_size // 2 - 0.5, grid_size // 2 - 0.5)
    
    def generate_ellipse(grid, r_x, r_y, angle):
        angle_rad = math.radians(angle)
        for x in range(grid_size):
            for y in range(grid_size):
                x_trans = x - center[0]
                y_trans = y - center[1]

                x_rot = x_trans * math.cos(angle_rad) - y_trans * math.sin(angle_rad)
                y_rot = x_trans * math.sin(angle_rad) + y_trans * math.cos(angle_rad)

                dist = (x_rot**2 / r_x**2) + (y_rot**2 / r_y**2)

                if 1 - (border_width / min(r_x, r_y)) <= dist <= 1:
                    grid[x, y] = 1  # Border is fully opaque
                elif dist < 1 - (border_width / min(r_x, r_y)):
                    grid[x, y] = mid_value  # Inside is semi-transparent

    def generate_circle(grid, radius):
        for x in range(grid_size):
            for y in range(grid_size):
                dist = ((x - center[0])**2 + (y - center[1])**2) ** 0.5
                if radius - border_width <= dist <= radius:
                    grid[x, y] = 1  # Border is fully opaque
                elif dist < radius - border_width:
                    grid[x, y] = mid_value  # Inside is semi-transparent

    def generate_star(grid, num_points=5, outer_radius=15, inner_radius=7):
        theta = np.linspace(0, 2 * np.pi, num_points * 2 + 1)
        outer_vertices = [
            (
                center[0] + outer_radius * math.cos(angle), 
                center[1] + outer_radius * math.sin(angle)
            )
            for angle in theta[::2]
        ]
        inner_vertices = [
            (
                center[0] + inner_radius * math.cos(angle), 
                center[1] + inner_radius * math.sin(angle)
            )
            for angle in theta[1::2]
        ]
        vertices = [coord for pair in zip(outer_vertices, inner_vertices) for coord in pair]
        polygon_fill(grid, vertices)

    def generate_random_polygon(grid, num_vertices=6, radius=15):
        angles = sorted([random.uniform(0, 2 * np.pi) for _ in range(num_vertices)])
        vertices = [
            (
                center[0] + radius * math.cos(angle), 
                center[1] + radius * math.sin(angle)
            )
            for angle in angles
        ]
        polygon_fill(grid, vertices)

    def generate_rectangle(grid, width, height):
        top_left = (center[0] - width // 2, center[1] - height // 2)
        for x in range(grid_size):
            for y in range(grid_size):
                if top_left[0] <= x <= top_left[0] + width and top_left[1] <= y <= top_left[1] + height:
                    if (
                        x <= top_left[0] + border_width or 
                        x >= top_left[0] + width - border_width or 
                        y <= top_left[1] + border_width or 
                        y >= top_left[1] + height - border_width
                    ):
                        grid[x, y] = 1  # Border is fully opaque
                    else:
                        grid[x, y] = mid_value  # Inside is semi-transparent

    def generate_triangle(grid, size):
        vertices = [
            (center[0], center[1] - size // 2),  # Top vertex
            (center[0] - size // 2, center[1] + size // 2),  # Bottom-left vertex
            (center[0] + size // 2, center[1] + size // 2)  # Bottom-right vertex
        ]
        polygon_fill(grid, vertices)

    def generate_blob(grid, max_radius=15):
        num_points = 100  # More points make the blob smoother
        theta = np.linspace(0, 2 * np.pi, num_points)
        radii = [random.uniform(max_radius * 0.7, max_radius) for _ in range(num_points)]
        vertices = [
            (
                center[0] + radii[i] * math.cos(angle), 
                center[1] + radii[i] * math.sin(angle)
            )
            for i, angle in enumerate(theta)
        ]
        polygon_fill(grid, vertices)

    def polygon_fill(grid, vertices):
        """Fill the inside of a polygon using the even-odd rule."""
        from matplotlib.path import Path

        # Create a Path object
        poly_path = Path(vertices)
        
        # Iterate through grid and fill points inside the polygon
        for x in range(grid_size):
            for y in range(grid_size):
                if poly_path.contains_point((x, y)):
                    grid[x, y] = mid_value  # Inside is semi-transparent

    # Randomly select a shape type
    shape_type = random.choice(['ellipse', 'circle', 'star', 'polygon', 'triangle', 'blob'])
    
    # Generate a shape that fills the grid
    if shape_type == 'ellipse':
        r_x = (grid_size // 2) - 2  # Maximize size within grid
        r_y = (grid_size // 2) - 2
        angle = random.uniform(0, 360)
        generate_ellipse(grid, r_x, r_y, angle)

    elif shape_type == 'circle':
        radius = (grid_size // 2) - 2
        generate_circle(grid, radius=radius)

    elif shape_type == 'star':
        num_points = random.randint(5, 8)
        outer_radius = (grid_size // 2) - 2
        inner_radius = random.randint(outer_radius // 2, outer_radius - 3)
        generate_star(grid, num_points=num_points, outer_radius=outer_radius, inner_radius=inner_radius)

    elif shape_type == 'polygon':
        num_vertices = random.randint(5, 10)
        radius = (grid_size // 2) - 2
        generate_random_polygon(grid, num_vertices=num_vertices, radius=radius)

    elif shape_type == 'rectangle':
        width = grid_size - 4  # Max width
        height = grid_size - 4  # Max height
        generate_rectangle(grid, width=width, height=height)

    elif shape_type == 'triangle':
        size = grid_size - 4  # Max size
        generate_triangle(grid, size=size)

    elif shape_type == 'blob':
        max_radius = (grid_size // 2) - 2  # Maximize blob size
        generate_blob(grid, max_radius=max_radius)

    return grid

def  genera_mascara2( img_size, borde=100, n_particles=10, tamany=60, particle_type="basic"):
   # Define some data:
  N = img_size[0]
  M = img_size[1]

  new_mask = np.zeros((N, M))

  for part in range(n_particles):
    # define random position
    x=np.random.randint(borde, N-tamany)
    y=np.random.randint(borde, M-tamany)

    #chose particle type
    if particle_type=="basic":
      particle = generate_circle_particle(grid_size=tamany)
    elif particle_type=="ellipse":
      particle = generate_ellipse_particle(grid_size=tamany, angle= np.random.randint(0,360))
    elif particle_type== "randomshape":
      particle =generate_random_shape(grid_size=tamany, num_points=12, border_width=2)


    # add particle
    x_len=particle.shape[0]
    y_len=particle.shape[1]
    off_x=int(x_len/2)
    off_y=int(y_len/2)
    mask = particle > 0
    new_mask[x:x+x_len, y:y+y_len][mask] += particle[mask]

  #fix values between 0 and 1
  new_mask=np.clip(new_mask, 0, 1)
  return new_mask


