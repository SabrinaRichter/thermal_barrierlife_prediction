from PIL import Image
import numpy as np
import pandas as pd
import xarray as xr

def read_data(csv_file_path='../data/train-orig.csv', tiff_folder_path='../data/train/'):
    df_meta = pd.read_csv(csv_file_path)
    im_list = []
    for i,imid in enumerate(df_meta.Image_ID):
        im = Image.open(tiff_folder_path+str(imid)+'.tif')
        if np.array(im).shape[1]>1024:
            im_list.append(np.array(im))
        if np.array(im).shape[1]<=1024:
            im = im.resize((2048,2048))
            im_list.append(np.array(im))

    ds = xr.Dataset({'greyscale':xr.DataArray(np.array(im_list))})
    ds = ds.rename({'dim_0':'image_id', 'dim_1':'x', 'dim_2':'y'})
    ds['sample'] = 'image_id', df_meta.Sample
    ds['lifetime'] = 'image_id', df_meta.Lifetime
    ds['magnification'] = 'image_id', df_meta.Magnification
    ds['uncertainty'] = 'image_id', df_meta.Uncertainty
    ds['image_id'] = 'image_id', df_meta.Image_ID
    return ds

