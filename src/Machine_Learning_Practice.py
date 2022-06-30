#import libraries for analysing the interpolated data
from netCDF4 import Dataset
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from datetime import datetime
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import PIL
import warnings

#import libraries for using machine learning models
import pandas as pd
from tensorflow import keras
from keras.utils import plot_model


