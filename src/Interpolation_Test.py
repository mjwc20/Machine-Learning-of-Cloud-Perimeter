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


#store the file name in a variable
file_name='sgpinterpolatedsondeC1.c1.20180403.000030.nc'


#store the path to the file in a variable
file_path='/Users/mattc/Documents/Uni/MSc/Project/Data/Bath/'
#read in the interpolation data into a dataset
interpolated_data=Dataset(file_path + file_name,mode='r')

warnings.filterwarnings("ignore", category=DeprecationWarning) 

'''dimensions:
        time = UNLIMITED ; // (1440 currently)
        height = 332 ;
variables:
        int base_time ;
                base_time:string = "2018-04-03 00:00:00 0:00" ;
                base_time:long_name = "Base time in Epoch" ;
                base_time:units = "seconds since 1970-1-1 0:00:00 0:00" ;
                base_time:ancillary_variables = "time_offset" ;
        double time_offset(time) ;
                time_offset:long_name = "Time offset from base_time" ;
                time_offset:units = "seconds since 2018-04-03 00:00:00 0:00" ;
                time_offset:ancillary_variables = "base_time" ;
        double time(time) ;
                time:long_name = "Time offset from midnight" ;
                time:units = "seconds since 2018-04-03 00:00:00 0:00" ;
                time:calendar = "gregorian" ;
                time:standard_name = "time" ;
        float height(height) ;
                height:long_name = "Height" ;
                height:units = "km" ;
                height:comment = "Height represents km above mean sea level" ;
                height:standard_name = "altitude" ;
        float precip(time) ;
                precip:long_name = "Precipitation" ;
                precip:units = "mm" ;
                precip:missing_value = -9999.f ;
                precip:ancillary_variables = "qc_precip" ;
                precip:standard_name = "lwe_thickness_of_precipitation_amount" ;
        int qc_precip(time) ;
                qc_precip:long_name = "Quality check results on field: Precipitation" ;
                qc_precip:units = "unitless" ;
                qc_precip:description = "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests." ;
                qc_precip:flag_method = "bit" ;
                qc_precip:bit_1_description = "Not used" ;
                qc_precip:bit_1_assessment = "Bad" ;
                qc_precip:bit_2_description = "Not used" ;
                qc_precip:bit_2_assessment = "Bad" ;
                qc_precip:bit_3_description = "Data value not available in input file, data value has been set to missing_value." ;
                qc_precip:bit_3_assessment = "Bad" ;
        float temp(time, height) ;
                temp:long_name = "Temperature" ;
                temp:units = "degC" ;
                temp:standard_name = "air_temperature" ;
                temp:valid_min = -90.f ;
                temp:valid_max = 50.f ;
                temp:missing_value = -9999.f ;
                temp:ancillary_variables = "qc_temp source_temp" ;
        int qc_temp(time, height) ;
                qc_temp:long_name = "Quality check results on field: Temperature" ;
                qc_temp:units = "unitless" ;
                qc_temp:description = "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests." ;
                qc_temp:flag_method = "bit" ;
                qc_temp:bit_1_description = "Value is less than the valid_min." ;
                qc_temp:bit_1_assessment = "Indeterminate" ;
                qc_temp:bit_2_description = "Value is greater than the valid_max." ;
                qc_temp:bit_2_assessment = "Indeterminate" ;
                qc_temp:bit_3_description = "Data value not available in input file, data value has been set to missing_value." ;
                qc_temp:bit_3_assessment = "Bad" ;
        int source_temp(time, height) ;
                source_temp:long_name = "Source for field: Temperature" ;
                source_temp:units = "unitless" ;
                source_temp:description = "This field contains integer values which should be interpreted as listed." ;
                source_temp:flag_method = "integer" ;
                source_temp:flag_0_description = "griddedsonde.c0:temp" ;
                source_temp:flag_1_description = "interpolated" ;
                source_temp:missing_value = -9999 ;
        float rh(time, height) ;
                rh:long_name = "Relative humidity" ;
                rh:units = "%" ;
                rh:standard_name = "relative_humidity" ;
                rh:valid_min = 0.f ;
                rh:valid_max = 105.f ;
                rh:missing_value = -9999.f ;
                rh:ancillary_variables = "qc_rh source_rh" ;
        int qc_rh(time, height) ;
                qc_rh:long_name = "Quality check results on field: Relative humidity" ;
                qc_rh:units = "unitless" ;
                qc_rh:description = "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests." ;
                qc_rh:flag_method = "bit" ;
                qc_rh:bit_1_description = "Value is less than the valid_min." ;
                qc_rh:bit_1_assessment = "Indeterminate" ;
                qc_rh:bit_2_description = "Value is greater than the valid_max." ;
                qc_rh:bit_2_assessment = "Indeterminate" ;
                qc_rh:bit_3_description = "Data value not available in input file, data value has been set to missing_value." ;
                qc_rh:bit_3_assessment = "Bad" ;
        int source_rh(time, height) ;
                source_rh:long_name = "Source for field: Relative humidity" ;
                source_rh:units = "unitless" ;
                source_rh:description = "This field contains integer values which should be interpreted as listed." ;
                source_rh:flag_method = "integer" ;
                source_rh:flag_0_description = "griddedsonde.c0:rh" ;
                source_rh:flag_1_description = "interpolated" ;
                source_rh:missing_value = -9999 ;
        float vap_pres(time, height) ;
                vap_pres:long_name = "Vapor pressure" ;
                vap_pres:units = "kPa" ;
                vap_pres:standard_name = "water_vapor_partial_pressure_in_air" ;
                vap_pres:missing_value = -9999.f ;
                vap_pres:ancillary_variables = "qc_vap_pres" ;
        int qc_vap_pres(time, height) ;
                qc_vap_pres:long_name = "Quality check results on field: Vapor pressure" ;
                qc_vap_pres:units = "unitless" ;
                qc_vap_pres:description = "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests." ;
                qc_vap_pres:flag_method = "bit" ;
                qc_vap_pres:bit_1_description = "Not used" ;
                qc_vap_pres:bit_1_assessment = "Bad" ;
                qc_vap_pres:bit_2_description = "Not used" ;
                qc_vap_pres:bit_2_assessment = "Bad" ;
                qc_vap_pres:bit_3_description = "Data value not available in input file, data value has been set to missing_value." ;
                qc_vap_pres:bit_3_assessment = "Bad" ;
        float bar_pres(time, height) ;
                bar_pres:long_name = "Barometric pressure" ;
                bar_pres:units = "kPa" ;
                bar_pres:standard_name = "air_pressure" ;
                bar_pres:valid_min = 0.f ;
                bar_pres:valid_max = 110.f ;
                bar_pres:missing_value = -9999.f ;
                bar_pres:ancillary_variables = "qc_bar_pres source_bar_pres" ;
        int qc_bar_pres(time, height) ;
                qc_bar_pres:long_name = "Quality check results on field: Barometric pressure" ;
                qc_bar_pres:units = "unitless" ;
                qc_bar_pres:description = "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests." ;
                qc_bar_pres:flag_method = "bit" ;
                qc_bar_pres:bit_1_description = "Value is less than the valid_min." ;
                qc_bar_pres:bit_1_assessment = "Indeterminate" ;
                qc_bar_pres:bit_2_description = "Value is greater than the valid_max." ;
                qc_bar_pres:bit_2_assessment = "Indeterminate" ;
                qc_bar_pres:bit_3_description = "Data value not available in input file, data value has been set to missing_value." ;
                qc_bar_pres:bit_3_assessment = "Bad" ;
        int source_bar_pres(time, height) ;
                source_bar_pres:long_name = "Source for field: Barometric pressure" ;
                source_bar_pres:units = "unitless" ;
                source_bar_pres:description = "This field contains integer values which should be interpreted as listed." ;
                source_bar_pres:flag_method = "integer" ;
                source_bar_pres:flag_0_description = "griddedsonde.c0:bar_pres" ;
                source_bar_pres:flag_1_description = "interpolated" ;
                source_bar_pres:missing_value = -9999 ;
        float wspd(time, height) ;
                wspd:long_name = "Wind speed" ;
                wspd:units = "m s-1" ;
                wspd:standard_name = "wind_speed" ;
                wspd:valid_min = 0.f ;
                wspd:valid_max = 100.f ;
                wspd:missing_value = -9999.f ;
                wspd:ancillary_variables = "qc_wspd" ;
        int qc_wspd(time, height) ;
                qc_wspd:long_name = "Quality check results on field: Wind speed" ;
                qc_wspd:units = "unitless" ;
                qc_wspd:description = "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests." ;
                qc_wspd:flag_method = "bit" ;
                qc_wspd:bit_1_description = "Value is less than the valid_min." ;
                qc_wspd:bit_1_assessment = "Indeterminate" ;
                qc_wspd:bit_2_description = "Value is greater than the valid_max." ;
                qc_wspd:bit_2_assessment = "Indeterminate" ;
                qc_wspd:bit_3_description = "Data value not available in input file, data value has been set to missing_value." ;
                qc_wspd:bit_3_assessment = "Bad" ;
        float wdir(time, height) ;
                wdir:long_name = "Wind direction" ;
                wdir:units = "degree" ;
                wdir:standard_name = "wind_to_direction" ;
                wdir:valid_min = 0.f ;
                wdir:valid_max = 360.f ;
                wdir:missing_value = -9999.f ;
                wdir:ancillary_variables = "qc_wdir" ;
        int qc_wdir(time, height) ;
                qc_wdir:long_name = "Quality check results on field: Wind direction" ;
                qc_wdir:units = "unitless" ;
                qc_wdir:description = "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests." ;
                qc_wdir:flag_method = "bit" ;
                qc_wdir:bit_1_description = "Value is less than the valid_min." ;
                qc_wdir:bit_1_assessment = "Indeterminate" ;
                qc_wdir:bit_2_description = "Value is greater than the valid_max." ;
                qc_wdir:bit_2_assessment = "Indeterminate" ;
                qc_wdir:bit_3_description = "Data value not available in input file, data value has been set to missing_value." ;
                qc_wdir:bit_3_assessment = "Bad" ;
        float u_wind(time, height) ;
                u_wind:long_name = "Eastward wind component" ;
                u_wind:units = "m s-1" ;
                u_wind:valid_min = -75.f ;
                u_wind:valid_max = 75.f ;
                u_wind:missing_value = -9999.f ;
                u_wind:ancillary_variables = "qc_u_wind source_u_wind" ;
                u_wind:standard_name = "eastward_wind" ;
        int qc_u_wind(time, height) ;
                qc_u_wind:long_name = "Quality check results on field: Eastward wind component" ;
                qc_u_wind:units = "unitless" ;
                qc_u_wind:description = "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests." ;
                qc_u_wind:flag_method = "bit" ;
                qc_u_wind:bit_1_description = "Value is less than the valid_min." ;
                qc_u_wind:bit_1_assessment = "Indeterminate" ;
                qc_u_wind:bit_2_description = "Value is greater than the valid_max." ;
                qc_u_wind:bit_2_assessment = "Indeterminate" ;
                qc_u_wind:bit_3_description = "Data value not available in input file, data value has been set to missing_value." ;
                qc_u_wind:bit_3_assessment = "Bad" ;
        int source_u_wind(time, height) ;
                source_u_wind:long_name = "Source for field: Eastward wind component" ;
                source_u_wind:units = "unitless" ;
                source_u_wind:description = "This field contains integer values which should be interpreted as listed." ;
                source_u_wind:flag_method = "integer" ;
                source_u_wind:flag_0_description = "griddedsonde.c0:u_wind" ;
                source_u_wind:flag_1_description = "interpolated" ;
                source_u_wind:missing_value = -9999 ;
        float v_wind(time, height) ;
                v_wind:long_name = "Northward wind component" ;
                v_wind:units = "m s-1" ;
                v_wind:valid_min = -75.f ;
                v_wind:valid_max = 75.f ;
                v_wind:missing_value = -9999.f ;
                v_wind:ancillary_variables = "qc_v_wind source_v_wind" ;
                v_wind:standard_name = "northward_wind" ;
        int qc_v_wind(time, height) ;
                qc_v_wind:long_name = "Quality check results on field: Northward wind component" ;
                qc_v_wind:units = "unitless" ;
                qc_v_wind:description = "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests." ;
                qc_v_wind:flag_method = "bit" ;
                qc_v_wind:bit_1_description = "Value is less than the valid_min." ;
                qc_v_wind:bit_1_assessment = "Indeterminate" ;
                qc_v_wind:bit_2_description = "Value is greater than the valid_max." ;
                qc_v_wind:bit_2_assessment = "Indeterminate" ;
                qc_v_wind:bit_3_description = "Data value not available in input file, data value has been set to missing_value." ;
                qc_v_wind:bit_3_assessment = "Bad" ;
        int source_v_wind(time, height) ;
                source_v_wind:long_name = "Source for field: Northward wind component" ;
                source_v_wind:units = "unitless" ;
                source_v_wind:description = "This field contains integer values which should be interpreted as listed." ;
                source_v_wind:flag_method = "integer" ;
                source_v_wind:flag_0_description = "griddedsonde.c0:v_wind" ;
                source_v_wind:flag_1_description = "interpolated" ;
                source_v_wind:missing_value = -9999 ;
        float dp(time, height) ;
                dp:long_name = "Dewpoint temperature" ;
                dp:units = "degC" ;
                dp:valid_min = -110.f ;
                dp:valid_max = 50.f ;
                dp:missing_value = -9999.f ;
                dp:ancillary_variables = "qc_dp source_dp" ;
                dp:standard_name = "dew_point_temperature" ;
        int qc_dp(time, height) ;
                qc_dp:long_name = "Quality check results on field: Dewpoint temperature" ;
                qc_dp:units = "unitless" ;
                qc_dp:description = "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests." ;
                qc_dp:flag_method = "bit" ;
                qc_dp:bit_1_description = "Value is less than the valid_min." ;
                qc_dp:bit_1_assessment = "Indeterminate" ;
                qc_dp:bit_2_description = "Value is greater than the valid_max." ;
                qc_dp:bit_2_assessment = "Indeterminate" ;
                qc_dp:bit_3_description = "Data value not available in input file, data value has been set to missing_value." ;
                qc_dp:bit_3_assessment = "Bad" ;
        int source_dp(time, height) ;
                source_dp:long_name = "Source for field: Dewpoint temperature" ;
                source_dp:units = "unitless" ;
                source_dp:description = "This field contains integer values which should be interpreted as listed." ;
                source_dp:flag_method = "integer" ;
                source_dp:flag_0_description = "griddedsonde.c0:dp" ;
                source_dp:flag_1_description = "interpolated" ;
                source_dp:missing_value = -9999 ;
        float potential_temp(time, height) ;
                potential_temp:long_name = "Potential temperature" ;
                potential_temp:units = "K" ;
                potential_temp:missing_value = -9999.f ;
                potential_temp:ancillary_variables = "qc_potential_temp" ;
                potential_temp:standard_name = "air_potential_temperature" ;
        int qc_potential_temp(time, height) ;
                qc_potential_temp:long_name = "Quality check results on field: Potential temperature" ;
                qc_potential_temp:units = "unitless" ;
                qc_potential_temp:description = "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests." ;
                qc_potential_temp:flag_method = "bit" ;
                qc_potential_temp:bit_1_description = "Not used" ;
                qc_potential_temp:bit_1_assessment = "Bad" ;
                qc_potential_temp:bit_2_description = "Not used" ;
                qc_potential_temp:bit_2_assessment = "Bad" ;
                qc_potential_temp:bit_3_description = "Data value not available in input file, data value has been set to missing_value." ;
                qc_potential_temp:bit_3_assessment = "Bad" ;
        float sh(time, height) ;
                sh:long_name = "Specific humidity" ;
                sh:units = "g/g" ;
                sh:missing_value = -9999.f ;
                sh:ancillary_variables = "qc_sh" ;
                sh:standard_name = "specific_humidity" ;
        int qc_sh(time, height) ;
                qc_sh:long_name = "Quality check results on field: Specific humidity" ;
                qc_sh:units = "unitless" ;
                qc_sh:description = "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests." ;
                qc_sh:flag_method = "bit" ;
                qc_sh:bit_1_description = "Not used" ;
                qc_sh:bit_1_assessment = "Bad" ;
                qc_sh:bit_2_description = "Not used" ;
                qc_sh:bit_2_assessment = "Bad" ;
                qc_sh:bit_3_description = "Data value not available in input file, data value has been set to missing_value." ;
                qc_sh:bit_3_assessment = "Bad" ;
        float rh_scaled(time, height) ;
                rh_scaled:long_name = "Relative humidity scaled using MWR" ;
                rh_scaled:units = "%" ;
                rh_scaled:valid_min = 0.f ;
                rh_scaled:valid_max = 105.f ;
                rh_scaled:missing_value = -9999.f ;
                rh_scaled:ancillary_variables = "aqc_rh_scaled qc_rh_scaled vapor_source" ;
        int qc_rh_scaled(time, height) ;
                qc_rh_scaled:long_name = "Quality check results on field: Relative humidity scaled using MWR" ;
                qc_rh_scaled:units = "unitless" ;
                qc_rh_scaled:description = "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests." ;
                qc_rh_scaled:flag_method = "bit" ;
                qc_rh_scaled:bit_1_description = "Value is less than the valid_min." ;
                qc_rh_scaled:bit_1_assessment = "Indeterminate" ;
                qc_rh_scaled:bit_2_description = "Value is greater than the valid_max." ;
                qc_rh_scaled:bit_2_assessment = "Indeterminate" ;
                qc_rh_scaled:bit_3_description = "Data value not available in input file, data value has been set to missing_value." ;
                qc_rh_scaled:bit_3_assessment = "Bad" ;
        int aqc_rh_scaled(time, height) ;
                aqc_rh_scaled:long_name = "Ancillary quality check results on field: Relative humidity scaled using MWR" ;
                aqc_rh_scaled:units = "unitless" ;
                aqc_rh_scaled:description = "This field contains integer values indicating the results of QC test on the data. Non-zero integers indicate the QC condition given in the description for those integers; a value of 0 indicates the data has not failed any QC tests." ;
                aqc_rh_scaled:flag_method = "integer" ;
                aqc_rh_scaled:flag_1_description = "Scale factor less than valid min, so RH has been scaled by interpolation of nearest valid scale factors" ;
                aqc_rh_scaled:flag_1_assessment = "Indeterminate" ;
                aqc_rh_scaled:flag_2_description = "Scale factor greater than valid max, so RH has been scaled by interpolation of nearest valid scale factors" ;
                aqc_rh_scaled:flag_2_assessment = "Indeterminate" ;
                aqc_rh_scaled:flag_3_description = "Scale factor less than valid min, but interpolation of nearest scale factors failed, so RH has not been scaled" ;
                aqc_rh_scaled:flag_3_assessment = "Indeterminate" ;
                aqc_rh_scaled:flag_4_description = "Scale factor greater than valid max, but interpolation of nearest scale factors failed, so RH has not been scaled" ;
                aqc_rh_scaled:flag_4_assessment = "Indeterminate" ;
                aqc_rh_scaled:flag_5_description = "Vapor unavailable, so RH has not been scaled" ;
                aqc_rh_scaled:flag_5_assessment = "Indeterminate" ;
                aqc_rh_scaled:flag_6_description = "RH not available in input file, data value has been set to missing_value." ;
                aqc_rh_scaled:flag_6_assessment = "Bad" ;
        int vapor_source(time, height) ;
                vapor_source:long_name = "Source of the MWR data used to produce: rh_scaled" ;
                vapor_source:units = "unitless" ;
                vapor_source:description = "This field contains integer values which should be interpreted as listed." ;
                vapor_source:flag_method = "integer" ;
                vapor_source:flag_0_description = "mwrret1liljclou.c1:be_pwv" ;
                vapor_source:flag_1_description = "mwrret1liljclou.c1:be_pwv interpolated" ;
                vapor_source:flag_2_description = "mwrlos.b1:vap" ;
                vapor_source:flag_3_description = "mwrlos.b1:vap interpolated" ;
                vapor_source:flag_4_description = "No vapor data" ;
        float lat ;
                lat:long_name = "North latitude" ;
                lat:units = "degree_N" ;
                lat:valid_min = -90.f ;
                lat:valid_max = 90.f ;
                lat:standard_name = "latitude" ;
        float lon ;
                lon:long_name = "East longitude" ;
                lon:units = "degree_E" ;
                lon:valid_min = -180.f ;
                lon:valid_max = 180.f ;
                lon:standard_name = "longitude" ;
        float alt ;
                alt:long_name = "Altitude above mean sea level" ;
                alt:units = "m" ;
                alt:standard_name = "altitude" ;

// global attributes:
                :command_line = "idl -D 0 -R -n interpolatedsonde -s sgp -f C1 -b 20180403 -P -n interpolatedsonde" ;
                :Conventions = "ARM-1.1" ;
                :process_version = "vap-interpolatedsonde-6.6-0.el6" ;
                :input_datastreams = "sgpgriddedsondeC1.c0 : 3.0 : 20180401.000030-20180405.000030\n",
                        "sgpmwrret1liljclouC1.c1 : 1.77 : 20180401.000035-20180405.000037\n",
                        "sgpmetE13.b1 : 4.37 : 20180401.000000-20180405.000000" ;
                :dod_version = "interpolatedsonde-c1-4.0" ;
                :site_id = "sgp" ;
                :platform_id = "interpolatedsonde" ;
                :facility_id = "C1" ;
                :data_level = "c1" ;
                :location_description = "Southern Great Plains (SGP), Lamont, Oklahoma" ;
                :datastream = "sgpinterpolatedsondeC1.c1" ;
                :doi = "10.5439/1095316" ;
                :history = "created by user dsmgr on machine ruby at 2018-04-11 18:03:22, using vap-interpolatedsonde-6.6-0.el6" ;
}'''

#cast the net cdf variables as numpy arrays to use in our plots
base_time=np.array(interpolated_data.variables['base_time'])

time_offset=np.array(interpolated_data.variables['time_offset'])

time=np.array(interpolated_data.variables['time'])

u_wind=np.array(interpolated_data.variables['u_wind'])

qc_u_wind=np.array(interpolated_data.variables['qc_u_wind'])

v_wind=np.array(interpolated_data.variables['v_wind'])

qc_v_wind=np.array(interpolated_data.variables['qc_v_wind'])

height = np.array(interpolated_data.variables['height'])

#turn the warnings back on
warnings.filterwarnings("default", category=DeprecationWarning) 

image_number=0
print(len(time))
for t in range(0,len(time),20):
    image_number=image_number+1

    height_indices_at_time_t = np.where((qc_u_wind[t,:]<=0 )& (qc_v_wind[t,:]<=0))

    plot_title = 'Image: ' + str(image_number) + ' '
    plot_title = plot_title + datetime.fromtimestamp(base_time+time_offset[t]).strftime("%B %d, %Y %I:%M:%S")
    plot_title = plot_title + '\n Eastward and Northward Wind Components'

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.set_title(plot_title, fontweight ='bold')
    ax.set_xlabel('Eastward Wind Component (m/s)')
    ax.set_ylabel('Northward Wind Component (m/s)')
    ax.set_zlabel('Height (km)')
    ax.set_xlim(-50,50)
    ax.set_zlim(0,30)
    ax.set_ylim(-50,50)
    ax.scatter3D(u_wind[t,height_indices_at_time_t],v_wind[t,height_indices_at_time_t],height[height_indices_at_time_t],c=height[height_indices_at_time_t])


    fig_name_base = '/Users/mattc/Documents/Uni/MSc/Project/VS Code/Machine-Learning-of-Cloud-Perimeter/Interpolation/'
    fig_name = fig_name_base  + str(image_number) + '.jpg'
    plt.savefig(fig_name)

    plt.close()

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
