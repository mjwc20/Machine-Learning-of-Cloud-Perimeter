# working atmospheric alignment
def atmospheric_generator_simple():
    # initialise a counter for the number of files considered
    counter = 0
    # store the path to the COGS files as a variable
    directory = "/content/gdrive/MyDrive/Colab Notebooks/Data/Bath/COGS/"
    # specify the folder containing the atmospheric files
    atmospheric_directory = "/content/gdrive/MyDrive/Colab Notebooks/Data/Bath/"
    # specify the folder containing the cloud perimeter value counter files
    counter_directory = "/content/gdrive/MyDrive/Colab Notebooks/Data/Variables/Counter/"
    # store the file path for the list of all COGS files in chronological order
    file_list_csv = "/content/gdrive/MyDrive/Colab Notebooks/Data/File_list.csv"
    # use pandas to read in the file list csv file and iterate through them
    df = pd.read_csv(file_list_csv)
    for index, filename in df.iterrows():
        print(filename['File_names'])
        # access each file name in the list using the key 'File_names'
        filename_str = str(filename['File_names'])
        # print the date as an error check
        print(filename_str[16:-12])
        # find the name of the corresponding atmospheric file
        atmospheric_filename = 'sgpinterpolatedsondeC1.c1.' + \
            filename_str[16:-12]+'.000030.nc'
        # specify the path to the atmospheric file
        atmospheric_file_path = atmospheric_directory + atmospheric_filename
        # print the file path to check it has been formatted correctly
        print(atmospheric_file_path)
        # read in the netCDF data as a Dataset
        interpolated_data = Dataset(atmospheric_file_path, mode='r')
        # store the path to the current cloud file as a variable
        file_path = directory+filename_str[1:-2]
        # read in the netCDF data as a Dataset
        cloud_data_set = Dataset(file_path, mode='r')
        # access the counter file from the directory and store as a dataframe
        perimeter_counter_file_path = counter_directory + \
            "Counter_"+filename_str[1:-2]
        counter_df = pd.read_csv(perimeter_counter_file_path)
        # then convert to numpy array of integers
        counter_df = counter_df.to_numpy(dtype='i')
        # add one to the counter as all new files have been accessed
        counter += 1
        # suppress the warnings that arise when casting as a numpy array
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # store the different cloud variables as numpy arrays
        #x = np.array(cloud_data_set.variables['x'])
        #y = np.array(cloud_data_set.variables['y'])
        #z = np.array(cloud_data_set.variables['z'])
        #cloud = np.array(cloud_data_set.variables['cloud'])
        cloud_time = np.array(cloud_data_set.variables['time'])
        # cast the net cdf atmospheric variables as numpy arrays
        base_time = np.array(interpolated_data.variables['base_time'])
        time_offset = np.array(interpolated_data.variables['time_offset'])
        interpolated_time = np.array(interpolated_data.variables['time'])
        height = np.array(interpolated_data.variables['height'])
        u_wind = np.array(interpolated_data.variables['u_wind'])
        qc_u_wind = np.array(interpolated_data.variables['qc_u_wind'])
        v_wind = np.array(interpolated_data.variables['v_wind'])
        qc_v_wind = np.array(interpolated_data.variables['qc_v_wind'])
        precip = np.array(interpolated_data.variables['precip'])
        qc_precip = np.array(interpolated_data.variables['qc_precip'])
        temp = np.array(interpolated_data.variables['temp'])
        qc_temp = np.array(interpolated_data.variables['qc_temp'])
        bar_pres = np.array(interpolated_data.variables['bar_pres'])
        qc_bar_pres = np.array(interpolated_data.variables['qc_bar_pres'])
        rh_scaled = np.array(interpolated_data.variables['rh_scaled'])
        qc_rh_scaled = np.array(interpolated_data.variables['qc_rh_scaled'])
        potential_temp = np.array(
            interpolated_data.variables['potential_temp'])
        qc_potential_temp = np.array(
            interpolated_data.variables['qc_potential_temp'])
        sh = np.array(interpolated_data.variables['sh'])
        qc_sh = np.array(interpolated_data.variables['qc_sh'])
        wdir = np.array(interpolated_data.variables['wdir'])
        qc_wdir = np.array(interpolated_data.variables['qc_wdir'])
        vap_pres = np.array(interpolated_data.variables['vap_pres'])
        qc_vap_pres = np.array(interpolated_data.variables['qc_vap_pres'])
        dp = np.array(interpolated_data.variables['dp'])
        qc_dp = np.array(interpolated_data.variables['qc_dp'])
        wspd = np.array(interpolated_data.variables['wspd'])
        qc_wspd = np.array(interpolated_data.variables['qc_wspd'])
        # turn deprecation warnings back on
        warnings.filterwarnings("default", category=DeprecationWarning)

        '''#replace all values which are not quality with the mean of all quality readings
    #THIS MAY NOT BE THE BEST STRATEGY BUT SHOULD ESSENTIALLY TELL THE NEURAL NET
    #TO DO THE USUAL THING WITH THAT VARIABLE
    wspd[qc_wspd>=1]=np.mean(wspd[qc_wspd<1])
    u_wind[qc_u_wind>=1]=np.mean(u_wind[qc_u_wind<1])
    v_wind[qc_v_wind>=1]=np.mean(v_wind[qc_v_wind<1])
    temp[qc_temp>=1]=np.mean(temp[qc_temp<1])
    bar_pres[qc_bar_pres>=1]=np.mean(bar_pres[qc_bar_pres<1])
    rh_scaled[qc_rh_scaled>=1]=np.mean(rh_scaled[qc_rh_scaled<1])
    potential_temp[qc_potential_temp>=1]=np.mean(potential_temp[qc_potential_temp<1])
    sh[qc_sh>=1]=np.mean(sh[qc_sh<1])
    wdir[qc_wdir>=1]=np.mean(wdir[qc_wdir<1])
    vap_pres[qc_vap_pres>=1]=np.mean(vap_pres[qc_vap_pres<1])
    dp[qc_dp>=1]=np.mean(dp[qc_dp<1])'''

        wspd_invalid = wspd[qc_wspd >= 1]
        u_wind_invalid = u_wind[qc_u_wind >= 1]
        v_wind_invalid = v_wind[qc_v_wind >= 1]
        temp_invalid = temp[qc_temp >= 1]
        bar_pres_invalid = bar_pres[qc_bar_pres >= 1]
        rh_scaled_invalid = rh_scaled[qc_rh_scaled >= 1]
        potential_temp_invalid = potential_temp[qc_potential_temp >= 1]
        sh_invalid = sh[qc_sh >= 1]
        wdir_invalid = wdir[qc_wdir >= 1]
        vap_pres_invalid = vap_pres[qc_vap_pres >= 1]
        dp_invalid = dp[qc_dp >= 1]

        wspd = fill(wspd, invalid=wspd_invalid)
        u_wind = fill(u_wind, invalid=u_wind_invalid)
        v_wind = fill(v_wind, invalid=v_wind_invalid)
        temp = fill(temp, invalid=temp_invalid)
        bar_pres = fill(bar_pres, invalid=bar_pres_invalid)
        rh_scaled = fill(rh_scaled, invalid=rh_scaled_invalid)
        potential_temp = fill(potential_temp, invalid=potential_temp_invalid)
        sh = fill(sh, invalid=sh_invalid)
        wdir = fill(wdir, invalid=wdir_invalid)
        vap_pres = fill(vap_pres, invalid=vap_pres_invalid)
        dp = fill(dp, invalid=dp_invalid)

        # normalise the variables ready for the neural network
        rh_scaled = (rh_scaled - np.mean(rh_scaled)) / np.std(rh_scaled)
        wspd = (wspd - np.mean(wspd)) / np.std(wspd)
        u_wind = (u_wind - np.mean(u_wind)) / np.std(u_wind)
        v_wind = (v_wind - np.mean(v_wind)) / np.std(v_wind)
        temp = (temp - np.mean(temp)) / np.std(temp)
        bar_pres = (bar_pres - np.mean(bar_pres)) / np.std(bar_pres)
        potential_temp = (
            potential_temp - np.mean(potential_temp)) / np.std(potential_temp)
        sh = (sh - np.mean(sh)) / np.std(sh)
        wdir = (wdir - np.mean(wdir)) / np.std(wdir)
        vap_pres = (vap_pres - np.mean(vap_pres)) / np.std(vap_pres)
        dp = (dp - np.mean(dp)) / np.std(dp)

        for i in range(len(cloud_time)):
            assert len(cloud_time) == len(
                counter_df), "mismatch between number of time steps in cloud_time and timesteps in the number of perimeter values vector"
            num_perimeter_values = int(counter_df[i])
            index = (cloud_time[i] - int(base_time)) // 60
            if (cloud_time[i] % 60) == 0:
                index = index - 1
            # find the height dimension
            height_dimension = np.shape(u_wind)[1]
            # initialise an np array
            atmospheric_variables_array = np.zeros((11, int(height_dimension)))
            atmospheric_variables = np.zeros((11, int(height_dimension)))
            atmospheric_variables[0, :] = wspd[index, :]
            atmospheric_variables[1, :] = u_wind[index, :]
            atmospheric_variables[2, :] = v_wind[index, :]
            atmospheric_variables[3, :] = temp[index, :]
            atmospheric_variables[4, :] = bar_pres[index, :]
            atmospheric_variables[5, :] = rh_scaled[index, :]
            atmospheric_variables[6, :] = potential_temp[index, :]
            atmospheric_variables[7, :] = sh[index, :]
            atmospheric_variables[8, :] = wdir[index, :]
            atmospheric_variables[9, :] = vap_pres[index, :]
            atmospheric_variables[10, :] = dp[index, :]
            for k in range(num_perimeter_values):
                np.stack((atmospheric_variables_array,
                         atmospheric_variables), axis=2)
        atmospheric_variables_array = atmospheric_variables_array[:, :, 1:]

        new_file_path_atmospheric = "/content/gdrive/MyDrive/Colab Notebooks/Data/Variables/Atmospheric/"
        new_file_name_atmospheric = new_file_path_atmospheric + \
            "Atmospheric_"+filename_str[1:-5]+".npy"
        np.save(new_file_name_atmospheric, atmospheric_variables_array)

        print(new_file_name_atmospheric)
        print('Counter='+str(counter))
        yield counter
