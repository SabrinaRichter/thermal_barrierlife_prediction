from PIL import Image
import numpy as np
import pandas as pd
import glob


def read_data(csv_file_path=False, tiff_folder_path='../data/train/'):
    if csv_file_path == False:
        fn_list = glob.glob(tiff_folder_path + '*.tif')
        im_list = []
        for fn in fn_list:
            im = Image.open(fn)
            if np.array(im).shape == (2048, 2048):
                im_list.append(np.array(im))
            else:
                im = im.resize((2048, 2048))
                im_list.append(np.array(im))

        # ds = build_data_xarray(images=np.array(im_list),
        #                        samples=np.full_like(np.array(fn_list), np.nan),
        #                        lifetimes=np.full_like(np.array(fn_list), np.nan),
        #                        maginifications=np.full_like(np.array(fn_list), np.nan),
        #                        uncertainities=np.full_like(np.array(fn_list), np.nan),
        #                        image_ids=np.array([s.split('/')[-1].split('.')[0] for s in fn_list]),
        #                        real_ids=np.full_like(np.array(fn_list), 1)
        #                        )
        ds = dict(greyscale=np.array(im_list),
                  sample=np.full_like(np.array(fn_list), np.nan),
                  lifetime=np.full_like(np.array(fn_list), np.nan),
                  magnification=np.full_like(np.array(fn_list), np.nan),
                  uncertainty=np.full_like(np.array(fn_list), np.nan),
                  image_id=np.array([s.split('/')[-1].split('.')[0] for s in fn_list]),
                  real=np.full_like(np.array(fn_list), 1))

    else:
        df_meta = pd.read_csv(csv_file_path)
        im_list = []
        for imid in df_meta.Image_ID:
            im = Image.open(tiff_folder_path + str(imid) + '.tif')
            if np.array(im).shape == (2048, 2048):
                im_list.append(np.array(im))
            else:
                im = im.resize((2048, 2048))
                im_list.append(np.array(im))

        # ds = build_data_xarray(images=np.array(im_list),
        #                        samples=df_meta.Sample,
        #                        lifetimes=df_meta.Lifetime,
        #                        maginifications=df_meta.Magnification,
        #                        uncertainities=df_meta.Uncertainty,
        #                        image_ids=df_meta.Image_ID,
        #                        real_ids=np.full_like(np.array(df_meta.Sample.values), 1),
        #                        )

        ds = dict(greyscale=np.array(im_list),
                  sample=df_meta.Sample.values.ravel(),
                  lifetime=df_meta.Lifetime.values.ravel(),
                  magnification=df_meta.Magnification.values.ravel(),
                  uncertainty=df_meta.Uncertainty.values.ravel(),
                  image_id=df_meta.Image_ID.values.ravel(),
                  real=np.full_like(np.array(df_meta.Sample.values.ravel()), 1))
    return ds


# def build_data_xarray(images: np.array, samples, lifetimes, maginifications, uncertainities, image_ids, real_ids):
#     """
#     Build dataset for multiple samples with list of images and list of metadata values for each variable
#     """
#     ds = xr.Dataset({'greyscale': xr.DataArray(images)})
#     ds = ds.rename({'dim_0': 'image_id', 'dim_1': 'x', 'dim_2': 'y'})
#     ds['sample'] = 'image_id', samples
#     ds['lifetime'] = 'image_id', lifetimes
#     ds['magnification'] = 'image_id', maginifications
#     ds['uncertainty'] = 'image_id', uncertainities
#     ds['image_id'] = 'image_id', image_ids
#     ds['real'] = 'image_id', real_ids
#
#     return ds
