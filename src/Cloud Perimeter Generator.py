from netCDF4 import Dataset
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import PIL
import warnings
from csv import writer
from scipy import ndimage as nd


#working perimeter computer (with corrected padding)
def file_generator_padding():
      #initialise a counter for the number of files considered
      counter=0
      #store the path to the COGS files as a variable
      directory = "/content/gdrive/MyDrive/Colab Notebooks/Data/Bath/COGS/"
      #store the file path for the list of all COGS files in chronological order
      file_list_csv = "/content/gdrive/MyDrive/Colab Notebooks/Data/processing_file_list.csv"
      #use pandas to read in the file list csv file and iterate through them
      df = pd.read_csv(file_list_csv)
      for row_index, filename in df.iterrows():
        #initialise a counter which counts the number of times a missing value occurs in a 
        #cell and that cell is therefore discounted  
        missing_value_counter=0
        #initialise the vectors which will store our variables cloud perimeter, cloud fraction
        #and height, these vectors will be the response and explanatory variables for the cells,
        #respectively
        cloud_perimeter_vector=[]
        cloud_fraction_vector=[]
        height_vector=[]
        perimeter_values_vector=[]
        #initialise the vector which stores the dimensions of the datasets
        dimensions=[]
        #print each file name as an error check and to monitor progress of the run
        print(filename['File_names'])
        #access each file name in the list using the key 'File_names'
        filename_str=str(filename['File_names'])
        #store the path to the current file as a variable
        file_path=directory+filename_str[1:-2]
        #add one to the counter as a new file has been accessed
        counter+=1
        #print the file path to check it has been formatted correctly
        print(file_path)
        #read in the netCDF data as a Dataset
        cloud_data_set = Dataset(file_path, mode='r')
        #suppress the warnings that arise when casting as a numpy array
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        #store the different cloud variables as numpy arrays
        x = np.array(cloud_data_set.variables['x'])
        y = np.array(cloud_data_set.variables['y'])
        z = np.array(cloud_data_set.variables['z'])
        cloud = np.array(cloud_data_set.variables['cloud'])
        time = np.array(cloud_data_set.variables['time'])
        #turn deprecation warnings back on
        warnings.filterwarnings("default", category=DeprecationWarning)
        #append the dimensions of the dataset i.e. the total possible cloud 
        #perimeter values to the end of the dimensions vector
        possible_values=np.shape(cloud)[0]*20*4*4
        #run through all time step in the file
        for t in range(cloud.shape[0]):
          #x and y dimensions have length 121, so only consider the final 120
          #values to make the cloud domain a cube
          cloud_cube=cloud
          cloud_slice=cloud_cube[t,:,:,:]
          perimeter_values_counter=0
          #specify the kernel which allows the computation of the cloud perimeter
          k=[[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,0,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]]
          #cast the cloud array and kernels as a Tensorflow tensor
          cloud_tensor = tf.constant(cloud_slice, tf.float32)
          k_tensor = tf.constant(k, tf.float32)
          #perform the convolution operation to find an array showing the number of neighbouring 1s for each element
          kernel_cloud = tf.nn.convolution(tf.reshape(cloud_tensor, [1, 120, 121, 121, 1]), tf.reshape(k_tensor, [3, 3, 3, 1, 1]), padding='VALID')
          #subtract this array from 6 in every entry to find the perimeter of each element
          kernel_cloud_subtracted = tf.subtract(6,kernel_cloud)
          #remove the outer edges of the cloud array so that we can multiply by the convolved arrya entry-wise
          #this has edges removed because VALID padding is used
          cloud_tensor_reduced=cloud_tensor[1:-1,1:-1,1:-1]
          #multiply the two arrays entry-wise so that we only consider the perimeter of elements where a cloud i.e. a 1
          #exists (where a 0 exists, this entry will just be zeroed by the operation)
          cloud_perimeter_tensor=tf.math.multiply(cloud_tensor_reduced,tf.squeeze(kernel_cloud_subtracted))
          #pad with an extra row of zeros to make the domain a cubic domain of 120 entries in each dimension
          for kk in range(20):
              for jj in range(4):
                  for ii in range(4):
                      if (kk==20 and jj==4 and ii==4):
                          #when we lie in the final cell we overlap by as much as necessary
                          cloud_cell=cloud_tensor_reduced[-6:,-30:,-30:]
                          cloud_perimeter_cell=cloud_perimeter_tensor[-6:,-30:,-30:]
                      elif (kk==20 and jj!=4 and ii!=4):
                          #when we lie in the final cell we overlap by as much as necessary
                          cloud_cell=cloud_tensor_reduced[-6:,30*(jj):30*(jj+1),30*(ii):30*(ii+1)]
                          cloud_perimeter_cell=cloud_perimeter_tensor[-6:,30*(jj):30*(jj+1),30*(ii):30*(ii+1)]
                      elif (kk==20 and jj==4 and ii!=4):
                          #when we lie in the final cell we overlap by as much as necessary
                          cloud_cell=cloud_tensor_reduced[-6:,-30:,30*(ii):30*(ii+1)]
                          cloud_perimeter_cell=cloud_perimeter_tensor[-6:,-30:,30*(ii):30*(ii+1)]
                      elif (kk==20 and jj!=4 and ii==4):
                          #when we lie in the final cell we overlap by as much as necessary
                          cloud_cell=cloud_tensor_reduced[-6:,30*(jj):30*(jj+1),-30:]
                          cloud_perimeter_cell=cloud_perimeter_tensor[-6:,30*(jj):30*(jj+1),-30:]
                      elif (kk!=20 and jj!=4 and ii==4):
                          #when we lie in the final cell we overlap by as much as necessary
                          cloud_cell=cloud_tensor_reduced[6*kk:6*(kk+1),30*(jj):30*(jj+1),-30:]
                          cloud_perimeter_cell=cloud_perimeter_tensor[6*kk:6*(kk+1),30*(jj):30*(jj+1),-30:]
                      elif (kk!=20 and jj==4 and ii==4):
                          #when we lie in the final cell we overlap by as much as necessary
                          cloud_cell=cloud_tensor_reduced[6*kk:6*(kk+1),-30:,-30:]
                          cloud_perimeter_cell=cloud_perimeter_tensor[6*kk:6*(kk+1),-30:,-30:]
                      elif (kk!=20 and jj==4 and ii!=4):
                          #when we lie in the final cell we overlap by as much as necessary
                          cloud_cell=cloud_tensor_reduced[6*kk:6*(kk+1),-30:,30*(ii):30*(ii+1)]
                          cloud_perimeter_cell=cloud_perimeter_tensor[6*kk:6*(kk+1),-30:,30*(ii):30*(ii+1)]
                      else:
                          #specify which of the 320 cells at each time step that we lie in
                          cloud_cell=cloud_tensor_reduced[6*kk:6*(kk+1),30*(jj):30*(jj+1),30*(ii):30*(ii+1)]
                          #specify the equivalent cell for the cloud_perimeter_tensor generated above
                          cloud_perimeter_cell=cloud_perimeter_tensor[6*kk:6*(kk+1),30*(jj):30*(jj+1),30*(ii):30*(ii+1)]
                      #if any value within the cell is -1 we discount the cell entirely,
                      #as prescribed by the methodology
                      if np.any(cloud_cell < 0):
                          #increment the counter by 1 if we ignore a cell
                          missing_value_counter+=1
                          continue
                      else:
                          perimeter_values_counter+=1
                          #compute the cloud fraction - divide the number of 1s by the volume of a cell
                          cloud_fraction_value=(np.sum(cloud_cell))/(np.shape(cloud_cell)[0]*np.shape(cloud_cell)[1]*np.shape(cloud_cell)[2])
                          #append the computed value onto the cloud fraction vector 
                          cloud_fraction_vector.append(cloud_fraction_value)
                          #sum over the entire 3D array to find the total perimeter - a single scalar
                          cloud_perimeter_value=tf.reduce_sum(cloud_perimeter_cell)
                          #append the cloud perimeter value for this time step in this cell to vector
                          cloud_perimeter_vector.append(cloud_perimeter_value)

                          #append the height to the height vector
                          height_vector.append(kk)
          perimeter_values_vector.append(perimeter_values_counter)


        cloud_perimeter_vector=np.array(cloud_perimeter_vector)
        cloud_fraction_vector=np.array(cloud_fraction_vector)
        height_vector=np.array(height_vector)
        perimeter_values_vector=np.array(perimeter_values_vector)
        #print out the number of perimeter values, missing values and possible values as an error check
        new_file_path_perimeter="/content/gdrive/MyDrive/Colab Notebooks/Data/Variables/Cloud Perimeter/"
        new_file_path_fraction="/content/gdrive/MyDrive/Colab Notebooks/Data/Variables/Cloud Fraction/"
        new_file_path_height="/content/gdrive/MyDrive/Colab Notebooks/Data/Variables/Height/"
        new_file_path_counter="/content/gdrive/MyDrive/Colab Notebooks/Data/Variables/Counter/"
        new_file_name_perimeter=new_file_path_perimeter+"Perimeter_"+filename_str[1:-5]+".csv"
        new_file_name_fraction=new_file_path_fraction+"Fraction_"+filename_str[1:-5]+".csv"
        new_file_name_height=new_file_path_height+"Height_"+filename_str[1:-5]+".csv"
        new_file_name_counter=new_file_path_counter+"Counter_"+filename_str[1:-5]+".csv"
        np.savetxt(new_file_name_perimeter, cloud_perimeter_vector)
        np.savetxt(new_file_name_fraction, cloud_fraction_vector)
        np.savetxt(new_file_name_height, height_vector)
        np.savetxt(new_file_name_counter, perimeter_values_vector)


        df=df.drop(index=row_index)
        df.to_csv(file_list_csv)


        print(new_file_name_perimeter)
        print('Counter='+str(counter))
        yield counter

all_files_iterator=True
for i in file_generator_padding():
  if i>200:
    break