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

                im = im.crop([0,0,712,712])
                im = im.resize((2048,2048))
                im_list.append(np.array(im))

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
                im = im.crop([0,0,712,712])
                im = im.resize((2048,2048))
                im_list.append(np.array(im))

        ds = dict(greyscale=np.array(im_list),
                  sample=df_meta.Sample.values.ravel() if 'Sample' in df_meta.columns else None,
                  lifetime=df_meta.Lifetime.values.ravel() if 'Lifetime' in df_meta.columns else None,
                  magnification=df_meta.Magnification.values.ravel() if 'Magnification' in df_meta.columns else None,
                  uncertainty=df_meta.Uncertainty.values.ravel() if 'Uncertainty' in df_meta.columns else None,
                  image_id=df_meta.Image_ID.values.ravel() if 'Image_ID' in df_meta.columns else None,
                  real=np.full_like(np.array(df_meta.Sample.values.ravel()), 1))
    return ds
