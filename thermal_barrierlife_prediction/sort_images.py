import os
import shutil
import pandas as pd

path_data = '/Users/karin.hrovatin/Documents/H3HerbstHackathon/challenge/git/thermal_barrierlife_prediction/data/'

def mkdir_missing(path, clean_exist=True):
    if clean_exist and os.path.isdir(path):
        shutil.rmtree(path)
    if not os.path.isdir(path):
        os.mkdir(path)


path_sorted = path_data + 'train_sorted/'
mkdir_missing(path_sorted)

metadata = pd.read_table(path_data + 'train-orig.csv', sep=',')

for group, data in metadata.groupby('Lifetime'):
    path = path_sorted + str(group) + '/'
    mkdir_missing(path)
    for img in data.Image_ID.values:
        img = img + '.tif'
        shutil.copyfile(path_data + 'train/' + img, path + img)
