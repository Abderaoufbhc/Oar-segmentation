%matplotlib inline

import numpy as np # linear algebra
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
from skimage.transform import resize, rescale, rotate, pyramid_reduce
from scipy.io import loadmat
from skimage.io import imread, imsave, imshow
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Datasets folder 
INPUT_FOLDER = "CT_Slices/"
patients = os.listdir(INPUT_FOLDER)
patients.sort()
# Load the scans in given folder path
MIN_BOUND = -1000.0 
MAX_BOUND = 400.0
def crop(x):
    volume, mask = x
    v_shape = volume.shape
    m_shape = mask.shape 
    black_slices=[]
    for i in range(mask.shape[0]):
      j=np.amax(mask[i])
      if j==0:
        black_slices.append(i)
    
    x = [i for i in range(mask.shape[0]) if i not in black_slices]
    volume = np.stack([volume[i] for i in x])
    mask=np.stack([mask[i] for i in x])
    return volume, mask
def externalm(x):
    volume, mask = x
    volume=volume * mask
    return volume,mask
def resize_sample(x, size=256):
    volume, mask = x
    v_shape = volume.shape
    out_shape = (v_shape[0], size, size,1)
    mask = resize(
        mask,
        output_shape=out_shape,
        order=0,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    out_shape = out_shape 
    volume = resize(
        volume,
        output_shape=out_shape,
        order=2,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    return volume, mask
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path) if s.endswith('.dcm') and s.startswith("C")]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

        
    return slices
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

volumes = {}
masks = {}
Heart_mask= {}


Rt_lung= loadmat('Ct_images/Rt_lung.mat')
Heart=loadmat('Ct_images/Heart.mat')
Lf_lung= loadmat('Ct_images/Lf_lung.mat')

for patient in sorted(patients):
  path=str(INPUT_FOLDER) + str(patient) + "/"
  dicom_slices = load_scan(path)
  volume = get_pixels_hu(dicom_slices)
  volumes[patient]=volume

y=[11,27,45,51]
x = [i for i in range(52) if i not in y]
for j,patients in zip(x,sorted(patients)):
  mask_re=[]
  for i in range(Rt_lung["Rt_lung"][0][j][0][0].shape[2]):
    mask_re.append(Rt_lung["Rt_lung"][0][j][0][0][:,:,i])
    mask_Rt_lung = np.stack([s for s in mask_re])
  
  mask_le=[]
  for i in range(Lf_lung["Lf_lung"][0][j][0][0].shape[2]):
    mask_le.append(Lf_lung["Lf_lung"][0][j][0][0][:,:,i])
    mask_Lf_lung = np.stack([s for s in mask_le])
  mask_lung= np.maximum(mask_Lf_lung, mask_Rt_lung)
  masks[patients]=mask_lung


y=[11,27,45,51]
x = [i for i in range(52) if i not in y]
patients=sorted(volumes)
for j,patients in zip(x,patients):
  mask_Heart=[]
  for i in range(Heart["Heart"][0][j][0][0].shape[2]):
    mask_Heart.append(Heart["Heart"][0][j][0][0][:,:,i])
  mask_volume = np.stack([s for s in mask_Heart])
  Heart_mask[patients]=mask_volume

patients = sorted(volumes)

volumes = [(volumes[k], Heart_mask[k]) for k in patients]
images=volumes[0][0][20]
print(images.shape)
mask=volumes[0][1][20]
imshow(images)
plt.show()
imshow(mask)
plt.show()


print("crop {} volumes...")
volumes = [crop(v) for v in volumes]
print("normalizing {} volumes...")
volumes = [(normalize(v), m) for v, m in volumes]
print("resize_sample {} volumes...")
image_size=256
volumes = [resize_sample(v, size=image_size) for v in volumes]

volumes_list=[]
masks_list=[]


for i in range(len(volumes)):
  volumes_list.append(volumes[i][0])
  masks_list.append(volumes[i][1])



volumes_list=np.array(volumes_list)
masks_list=np.array(masks_list)


print(volumes_list.shape)
print(masks_list.shape)
image_list=[]
mask_list=[]
for volume_i in volumes_list:
  for image in volume_i:
    image_list.append(image)

for volume_m in masks_list:
  for mask_l in volume_m:
    mask_list.append(mask_l)


image_list=np.array(image_list)
mask_list=np.array(mask_list)
print(len(mask_list))

x_train, x_val, y_train, y_val = train_test_split(image_list, mask_list, test_size=0.1)

np.save('dataset_Heart/x_train.npy', x_train)
np.save('dataset_Heart/y_train.npy', y_train)
np.save('dataset_Heart/x_val.npy', x_val)
np.save('dataset_Heart/y_val.npy', y_val)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
