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

#interpolated cloud data netCDF file
file_name ='sgpinterpolatedsondeC1.c1.20170831.000030.nc'

#read in the cloud dataset
file_path = '/Users/mattc/Documents/Uni/MSc/Project/Data/Bath/'
interplated_data_set = Dataset(file_path + file_name, mode='r') 

#set the Boolean to True to print metadata for the cloud dataset
print_ds_metadata = True
if print_ds_metadata:
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    print(interplated_data_set)
    warnings.filterwarnings("default", category=DeprecationWarning) 
#note, to read metadata in the terminal type: ncdump -h file_name
#for example:
#ncdump -h 'C:\Users\adria\Documents\Programming\VS Code\sgpinterpolatedsondeC1.c1.20170831.000030.nc'

#extract variables from the netCDF file
#note that extracting variables as numpy arrays causes a deprication warning
#suppress the deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
#int base_time in seconds is the number of seconds since 1/1/1970
base_time = np.array(interplated_data_set.variables['base_time'])
#double time_offset in seconds since 1/1/1970 (time offset from base time)
time_offset = np.array(interplated_data_set.variables['time_offset'])
#double time in seconds since 31/8/2017 (time offset from midnight)
time = np.array(interplated_data_set.variables['time'])
#float height(height) in km above mean sea level
height = np.array(interplated_data_set.variables['height'])
#float precip(time) in mm is lwe thickness of precipitation amount, missing value -9999.f
precip = np.array(interplated_data_set.variables['precip'])
#precipitation quality check: each bit represents a QC test on the data. Non-zero bits indicate the QC condition 
#given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests
qc_precip = np.array(interplated_data_set.variables['qc_precip'])
#float temp(time, height) in degrees C is the air temperature; min -90.f and max 50.f, missing value -9999.f
temp = np.array(interplated_data_set.variables['temp'])
#temperature quality check: each bit represents a QC test on the data. Non-zero bits indicate the QC condition 
#given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests
qc_temp = np.array(interplated_data_set.variables['qc_temp'])
#int source_temp(time, height) is unitless source for temperature, integer flag 0 "griddedsonde.c0:temp" and flag 1 "interpolated", missing value -9999
#float rh(time, height) is the percentage relative humidity, min 0.f and max 105.f, missing value -9999.f
#int source_rh is unitless source of relative humidity, integer flag 0 "gridedseonde.c0:rh" and flag 1 "interpolated", missing value -9999
#float vap_pres(time, height) in kPa is the water vapour partial pressure in air, missing value -9999.f
#float bar_pres(time, height) in kPa is the air barometric air pressure, min 0.f and max 110.f, missing value -9999.f
bar_pres = np.array(interplated_data_set.variables['bar_pres'])
#barometric air pressure quality check: each bit represents a QC test on the data. Non-zero bits indicate the QC condition 
#given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests
qc_bar_pres = np.array(interplated_data_set.variables['qc_bar_pres'])
#int source_bar_pres is unitless source of Barometric pressure, integer flag 0 "gridedseonde.c0:bar_pres" and flag 1 "interpolated", missing value -9999
#float wspd(time, height) in m/s is the wind speed, min 0.f and max 100.f, missing value -9999.f
#float wdir(time, height) in degrees is the wind direction, min 0.f and max 360.f, missing value -9999.f
#float u_wind(time, height) in m/s is the Eastward wind component, min -75.f and max 75.f, missing -9999.f
#int source_u_wind(time, height) is the unitless source for Eastward wind component, integer flag 0 "gridedseonde.c0:u_wind" and flag 1 "interpolated", missing value -9999.f
#float v_wind(time, height) in m/s is the Northward wind component, min -75.f and max 75.f, missing -9999.f
#int source_v_wind(time, height) is the unitless source for Northward wind component, integer flag 0 "gridedseonde.c0:u_wind" and flag 1 "interpolated", missing value -9999.f
#float dp(time, height) in degrees C is the dewpoint temperature, min -110.f and max 50.f, missing value -9999.f
#int source_dp(time, height) is the unitless source for depoint temperature, integer flag 0 "gridedseonde.c0:dp" and flag 1 "interpolated", missing value -9999.f
#float potential_temp(time, height) in K is the potential temperature, missing value -9999.f
#float sh(time, height) is the specific humidity in g/g, missing value is -9999.f
#float rh_scaled(time, height) is the relative humidity scaled using MWR, min 0.f and max 105.f, missing value -9999.f
#int vapor_source(time, height) is the unitless source of relative humidity scaled, integer flag 0 "mwrret1liljclou.c1:be_pwv", flag 1 "mwrret1liljclou.c1:be_pwv interpolated", flag 2 "mwrlos.b1:vap", flag 3 "mwrlos.b1:vap interpolated", flag 4 "No vapor data" 
#float lat in degrees North is the North latitdue, min -90.f and max 90.f
#float lon in degrees East is the East latitdue, min -180.f and max 180.f
#float alt in m is the altitude above mean sea level

#turn deprecation warnings back on
warnings.filterwarnings("default", category=DeprecationWarning) 

#if required, print out sizes of the variables
if print_ds_metadata:
    print('base_time is: ' + datetime.fromtimestamp(base_time).strftime("%B %d, %Y %I:%M:%S"))
    print('time_offset shape is: ',np.shape(time_offset))
    print('time shape is: ',np.shape(time))
    #print('time[0] and [1] = ',time[0], time[1])
    #print('time_offset[0] and [1] = ',time_offset[0], time_offset[1])
    print('height shape is: ',np.shape(height))
    print('precip shape is: ',np.shape(precip))
    print('temp shape is: ',np.shape(temp))

#find matrix elements for actual temperatures, where it is a not missing value (-9999.f)
index_temp_actual=np.where(temp>=-90)

show_3d_plot = False
if show_3d_plot:
    plot_title = 'Temperature by Height & Secs after ' 
    plot_title = plot_title + datetime.fromtimestamp(base_time).strftime("%B %d, %Y %I:%M:%S")
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection ="3d")
    ax.set_title(plot_title, fontweight ='bold')
    ax.set_xlabel('Time (s)', fontweight ='bold')
    ax.set_ylabel('Height (km)', fontweight ='bold')
    ax.set_zlabel('Temperature (C)', fontweight ='bold')
    ax.scatter3D(time_offset[index_temp_actual[0]], height[index_temp_actual[1]], temp[index_temp_actual])
    plt.show()

show_color_plot = False
if show_color_plot:
    fig = plt.figure(figsize=(4, 8))
    ax = fig.add_subplot(111)
    ax.set_title('Temperature')
    #plt.imshow(temp)
    #plt.colorbar(orientation='vertical')
    plt.pcolormesh(temp)
    plt.show()

produce_timelapse = True
image_number=0
if produce_timelapse:
    for t in range(0, len(time), 20):
        image_number=image_number+1

        #find indices for actual temperatures at time t, where it is a not missing value (-9999.f)
        #index_temp_at_t=np.where(temp[t,:]>=-90)
        #find indices for the temperature data has not failed any quality checks
        index_temp_at_t=np.where(qc_temp[t,:]<=0)
        index_bar_pres_at_t=np.where(qc_bar_pres[t,:]<=0)

        plot_title = 'Image: ' + str(image_number) + ' '
        plot_title = plot_title + datetime.fromtimestamp(base_time+time_offset[t]).strftime("%B %d, %Y %I:%M:%S")
        plot_title = plot_title + '\n Temperature and Pressure' 
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.set_title(plot_title, fontweight ='bold')
        ax.set_xlabel('Height (km)', fontweight ='bold')
        ax.set_ylabel('Temperature (C)', fontweight ='bold')
        ax.set_xlim(0,28)
        ax.set_ylim(-70,100)
        ax.scatter(height[index_temp_at_t[0]],temp[t,index_temp_at_t[0]], c='blue', label='Temperature')
        ax2 = ax.twinx()
        ax2.set_ylabel('Barometric Air Pressure (kPa)', fontweight ='bold')
        ax2.set_ylim(-70,100)
        ax2.scatter(height[index_bar_pres_at_t[0]],bar_pres[t,index_bar_pres_at_t[0]], c='red', label='Barometric Air Pressure')
        #add a legend for both scatter plots
        marks, labels = ax.get_legend_handles_labels()
        marks2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(marks + marks2, labels + labels2, loc=0)

        #save the plot to file
        fig_name_base = '/Users/mattc/Documents/Uni/MSc/Project/VS Code/Machine-Learning-of-Cloud-Perimeter/Interpolation'
        fig_name = fig_name_base  + str(image_number) + '.jpg'
        plt.savefig(fig_name)
        #show the plot
        #plt.show()
        #close the plot to save memory
        plt.close()

    #create a timelapse for all the images
    image_frames = []
    for image_index in range(1,image_number+1):
        new_frame = PIL.Image.open(fig_name_base + str(image_index) + '.jpg') 
        image_frames.append(new_frame)
    # save as GIF
    image_frames[0].save(fig_name_base + 'timelapse.gif',
                format='GIF',
                append_images = image_frames[1: ],
                save_all = True, 
                duration = 250,
                loop = 0)