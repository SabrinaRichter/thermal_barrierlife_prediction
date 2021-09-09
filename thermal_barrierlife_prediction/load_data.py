from PIL import Image
import numpy as np
import pandas as pd
import xarray as xr
import glob

def read_data(csv_file_path=False, tiff_folder_path='../data/train/'):
    if csv_file_path == False:
        fn_list = glob.glob(tiff_folder_path+'*.tif')
        im_list = []
        for fn in fn_list:
            im = Image.open(fn)
            if np.array(im).shape[1]>1024:
                im_list.append(np.array(im))
            if np.array(im).shape[1]<=1024:
                im = im.resize((2048,2048))
                im_list.append(np.array(im))
        ds = xr.Dataset({'greyscale':xr.DataArray(np.array(im_list))})
        ds = ds.rename({'dim_0':'image_id', 'dim_1':'x', 'dim_2':'y'})
        ds['sample'] = 'image_id', np.full_like(np.array(fn_list), np.nan)
        ds['lifetime'] = 'image_id', np.full_like(np.array(fn_list), np.nan)
        ds['magnification'] = 'image_id', np.full_like(np.array(fn_list), np.nan)
        ds['uncertainty'] = 'image_id', np.full_like(np.array(fn_list), np.nan)
        ds['image_id'] = 'image_id', np.array([s.split('/')[-1].split('.')[0] for s in fn_list])
        
    else:
        df_meta = pd.read_csv(csv_file_path)
        im_list = []
        for imid in df_meta.Image_ID:
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

