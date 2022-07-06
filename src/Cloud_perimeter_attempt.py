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

#cloud data netCDF file
#file_name = 'sgpcogspiN1.c1.20170831.180000.nc'
file_name ='sgpcogspiN1.c1.20180206.143820.nc'

#read in the cloud dataset
file_path = '/Users/mattc/Documents/Uni/MSc/Project/VS Code/Machine-Learning-of-Cloud-Perimeter/'
cloud_data_set = Dataset(file_path + file_name, mode='r') 

#set the Boolean to True to print metadata for the cloud dataset
print_ds_metadata = False
if print_ds_metadata:
    print(cloud_data_set)
#note, to read metadata in the terminal type: ncdump -h file_name
#for example:
#ncdump -h '\Users\mattc\Documents\Uni\MSc\Project\VS Code\Machine-Learning-of-Cloud-Perimeter\sgpcogspiN1.c1.20180206.143820.nc'

#extract variables x,y,z,cloud and time as numpy arrays
#note that extracting variables as numpy arrays causes a deprication warning
#suppress the deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
x = np.array(cloud_data_set.variables['x'])
y = np.array(cloud_data_set.variables['y'])
z = np.array(cloud_data_set.variables['z'])
cloud = np.array(cloud_data_set.variables['cloud'])
time = np.array(cloud_data_set.variables['time'])
#turn deprecation warnings back on
warnings.filterwarnings("default", category=DeprecationWarning) 

#if required, print out sizes of the variables
if print_ds_metadata:
    print('x shape is: ',np.shape(x))
    print('y shape is: ',np.shape(y))
    print('z shape is: ',np.shape(z))
    print('time shape is: ',np.shape(time))
    print('cloud shape is: ',np.shape(cloud))

#set the range of time indices to include, with a stride if required
time_range = np.arange(0,time.size,1)
#time_range = np.arange(0,1,1)

#iterate through each time point and plot the cloud
plot_data=False
if plot_data:
    image_number=0
    for t_index in time_range:
        fig = plt.figure(figsize = (12, 10))
        ax = fig.add_subplot(2,2,1,projection='3d')
        #show the date and time as the plot title
        plot_title = 'Cloud Image: ' + str(image_number+1)
        plot_title = plot_title + ' ' + datetime.fromtimestamp(time[t_index]).strftime("%B %d, %Y %I:%M:%S")
        ax.set_title(plot_title, fontweight ='bold')
        #obtain the cloud matrix indices for positive elements (which correspond to positions with cloud)
        #1 => cloud, 0 => no cloud and -1 => no cloud reconstruction
        zt,yt,xt = np.where(cloud[t_index,:,:,:]>0)
        #compute the amount of cloud in each vertical direction (which contains some cloud)

        if xt.size>0:
            #only plot if there is some cloud
            image_number=image_number+1
            #set axes labels
            ax.set_xlabel('x', fontweight ='bold')
            ax.set_ylabel('y', fontweight ='bold')
            ax.set_zlabel('z', fontweight ='bold')
            #plot a 3d scatter for the positive elements
            #using values for x,y,z
            ax.scatter3D(x[xt], y[yt], z[zt], c=z[zt])
            ax.set_xlim(x[0],x[-1])
            ax.set_ylim(y[0],y[-1])
            ax.set_zlim(z[0],2000)
            
            #Second plot, viewed from above
            ax = fig.add_subplot(2,2,3,projection='3d')
            plot_indices=False
            ax.scatter3D(x[xt], y[yt], z[zt], c=z[zt])
            ax.set_xlim(x[0],x[-1])
            ax.set_ylim(y[0],y[-1])
            ax.set_zlim(z[0],2000)
            ax.set_title('View from Above', fontweight ='bold')
            #set the view angle
            ax.view_init(90, 0)
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.set_xlabel('x', fontweight ='bold')

            #Third plot, viewed from side
            ax = fig.add_subplot(2,2,2,projection='3d')
            plot_indices=False
            ax.scatter3D(x[xt], y[yt], z[zt], c=z[zt])
            ax.set_xlim(x[0],x[-1])
            ax.set_ylim(y[0],y[-1])
            ax.set_zlim(z[0],2000)
            ax.set_title('View of Y-Z Plane', fontweight ='bold')
            #set the view angle
            ax.view_init(0, 0)
            ax.set_xticklabels([])
            ax.set_ylabel('y', fontweight ='bold')
            ax.set_zlabel('z', fontweight ='bold')

            #Fourth plot, viewed from side
            ax = fig.add_subplot(2,2,4,projection='3d')
            plot_indices=False
            ax.scatter3D(x[xt], y[yt], z[zt], c=z[zt])
            ax.set_xlim(x[0],x[-1])
            ax.set_ylim(y[0],y[-1])
            ax.set_zlim(z[0],2000)
            ax.set_title('View of X-Z Plane', fontweight ='bold')
            #set the view angle
            ax.view_init(0, 90)
            ax.set_yticklabels([])
            ax.set_xlabel('x', fontweight ='bold')
            ax.set_zlabel('z', fontweight ='bold')

            #save the plot to file
            fig_name_base = '/Users/mattc/Documents/Uni/MSc/Project/VS Code/Machine-Learning-of-Cloud-Perimeter/Clouds/' 
            fig_name = fig_name_base + 'cloud_' + str(image_number) + '.jpg'
            plt.savefig(fig_name)
            #plt.pause(1)
            #close the plot to save memory
            plt.close()
        else:
            print('t_index = ',time(t_index))
            
    # Show plot
    #plt.show()

    #create a timelapse for all the images
    image_frames = []
    for image_index in range(1,image_number+1):
        new_frame = PIL.Image.open(fig_name_base + 'cloud_' + str(image_index) + '.jpg') 
        image_frames.append(new_frame)
    # save as GIF
    image_frames[0].save(fig_name_base + 'cloud_timelapse.gif',
                    format='GIF',
                    append_images = image_frames[1: ],
                    save_all = True, 
                    duration = 500,
                    loop = 0)


cloud_slice=cloud[18,:,:,:]
cloud_slice[cloud_slice==-1]=0

kernel_method=True
if kernel_method:
    k=[[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,0,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]]
    cloud_tensor = tf.constant(cloud_slice, tf.float32)
    k_tensor = tf.constant(k, tf.float32)
    kernel_cloud = tf.nn.convolution(tf.reshape(cloud_tensor, [1, 120, 121, 121, 1]), tf.reshape(k_tensor, [3, 3, 3, 1, 1]), padding='VALID')
    kernel_cloud_subtracted = tf.subtract(6,kernel_cloud)
    cloud_tensor_reduced=cloud_tensor[1:-1,1:-1,1:-1]
    cloud_perimeter_tensor=tf.math.multiply(cloud_tensor_reduced,tf.squeeze(kernel_cloud_subtracted))
    cloud_perimeter=tf.reduce_sum(cloud_perimeter_tensor)
    print(cloud_perimeter)