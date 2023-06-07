# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import scipy, scipy.signal
import matplotlib
from pylab import imshow
import h5py
import pickle

# %%
from jupytext.config import global_jupytext_configuration_directories
list(global_jupytext_configuration_directories())

# %%
import shutup, time, random

shutup.please()

import numpy as np
import pandas as pd
import sys, os, os.path, shutil

from datetime import date, datetime
from random import seed, random
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import scipy, scipy.signal
import pickle, h5py

# %%
VI_idx = "NDVI"
smooth = "SG"
batch_no = "1"
model = "DL"


# %%
def load_image(filename):
    img = load_img(filename, target_size=(224, 224))  # load the image
    img = img_to_array(img)  # convert to array
    img = img.reshape(1, 224, 224, 3)  # reshape into a single sample with 3 channels
    img = img.astype("float32")  # center pixel data
    img = img - [123.68, 116.779, 103.939]
    return img



# %%
winnerModels = pd.read_csv("/Users/hn/Documents/test_DL/" + "winnerModels.csv")

# %%
winnerModel = np.array(
    winnerModels.loc[
        (winnerModels.VI_idx == VI_idx)
        & (winnerModels.smooth == smooth)
        & (winnerModels.model == model)
    ].output_name
)[0]
print("winnerModel=", winnerModel)

# %%
winnerModel.endswith(".sav")

# %%

# %%

# %%
