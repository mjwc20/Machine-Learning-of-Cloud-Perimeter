import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


a=np.array([1,0,1,1])
b=np.array([1,0,0,1])
c=np.array([1,0,0,1])
indices = np.where(np.logical_and(a>=1,b>=1,c>=1))
print(indices)