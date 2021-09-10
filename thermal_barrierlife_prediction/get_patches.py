from thermal_barrierlife_prediction.load_data import read_data
import numpy as np
import random

def get_patches_tf(patch_width = 256, num_patches = 100):
    """
    """
    return 0

def get_patches( patch_width = 256, num_patches = 100):
    """
    input:
    ds: xarray data frame of the original whole images
    patch_width: width of the patches assuming square
    num_patches: number of patches per whole images
    
    return:
    patch_ds: xarray dataframe of the random patches
    """
    # load data
    ds = read_data('../data/train-orig.csv', '../data/train/')
   # data_frame = read_data()
    X = ds['greyscale'].to_numpy()
    Y = ds['lifetime'].to_numpy()
    U = ds['uncertainty'].to_numpy()

    # Specify variables
    image_shape_x = X.shape[1] # assuming the whole images are always square
    total_images  = X.shape[0] # number of whole images
    #patch_width   = 256                   # width of the squared patches
    #num_patches   = 100                   # number of random patches per whole image

    # initialize numpy array
    x_patches   = np.zeros((total_images*num_patches, patch_width, patch_width ))
    y_lifetime  = np.zeros((total_images*num_patches))
    y_uncert    = np.zeros((total_images*num_patches))
    
    
    for i in range(total_images):
        num_selected = 0
        while num_selected<  num_patches:
            random_x = random.randint(0, image_shape_x - patch_width)
            random_y = random.randint(0, image_shape_x - patch_width)
            x_patches[i*num_patches + num_selected,:,:] = X[i, random_x:random_x + patch_width, random_y:random_y + patch_width ]
            y_lifetime[i*num_patches + num_selected]    = Y[i]
            y_uncert[i*num_patches + num_selected]      = U[i]
            #generate random index
            num_selected +=1

    patch_ds = dict()
    patch_ds['patches'] = x_patches
    patch_ds['uncertainty'] = y_uncert
    patch_ds['lifetime'] = y_lifetime

    return patch_ds


# examples:
# from get_patches import get_patches
# patch_ds = get_patches(patch_width = 256, num_patches = 50 )
# patch_ds["lifetime"].shape
#