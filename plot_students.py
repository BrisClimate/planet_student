### This plots various metrics for the first year atmospheric science practical

import numpy as np
import xarray as xar
import matplotlib.pyplot as plt
import os as os

# Read in the environment variable as the path
path = os.environ['GFDL_DATA']

# Open control data
dataset_control = xar.open_dataset(path + '/frierson1/run0023/atmos_monthly.nc')
dataset_exp = xar.open_dataset(path + '/frierson2/run0023/atmos_monthly.nc')

#dataset['land_mask'].mean('lon').plot()
plt.figure('Zonal Wind Control')
dataset_control['ucomp'].mean('lon')[0,:,:].plot.contourf(cmap = 'bwr',yincrease=False,cbar_kwargs={'label': 'Westerly wind speed (m/s)'})
plt.ylabel('Pressure (hPa)')
plt.xlabel('Latitude')
plt.title('Zonal Wind (Earth)')

plt.figure('Zonal Wind Experiment')
dataset_exp['ucomp'].mean('lon')[0,:,:].plot.contourf(cmap = 'bwr',yincrease=False,cbar_kwargs={'label': 'Westerly wind speed (m/s)'})
plt.ylabel('Pressure (hPa)')
plt.xlabel('Latitude')
plt.title('Zonal Wind (Experiment)')

ucomp_diff = dataset_control['ucomp'].values - dataset_exp['ucomp'].values
lat =  dataset_control['lat'].values
pfull =  dataset_control['pfull'].values


plt.figure('Zonal Wind Difference (Earth minus Experiment)')
plt.contourf(lat,pfull,np.mean(ucomp_diff,axis=3)[0,:,:],cmap='bwr')
plt.gca().invert_yaxis()
plt.colorbar(label='Westerly wind speed (m/s)')
plt.title('Zonal Wind Difference (Earth minus Experiment)')
plt.ylabel('Pressure (hPa)')
plt.xlabel('Latitude')

# zonal temperature

plt.figure('Zonal Temperature (Earth)')
dataset_control['temp'].mean('lon')[0,:,:].plot.contourf(cmap = 'hot',yincrease=False,cbar_kwargs={'label': '(K)'})
plt.ylabel('Pressure (hPa)')
plt.xlabel('Latitude')

plt.figure('Zonal Temperature (Experiment)')
dataset_exp['temp'].mean('lon')[0,:,:].plot.contourf(cmap = 'hot',yincrease=False,cbar_kwargs={'label': '(K)'})
plt.ylabel('Pressure (hPa)')
plt.xlabel('Latitude')

tcomp_diff = dataset_control['temp'].values - dataset_exp['temp'].values
lat =  dataset_control['lat'].values
pfull =  dataset_control['pfull'].values

plt.figure('Zonal T Difference (Earth minus Experiment)')
plt.contourf(lat,pfull,np.mean(tcomp_diff,axis=3)[0,:,:],cmap='bwr')
plt.gca().invert_yaxis()
plt.colorbar(label='(K)')
plt.title('Zonal T Difference')
plt.ylabel('Pressure (hPa)')
plt.xlabel('Latitude')



# Meridional wind

plt.figure('Meridional Wind (Earth)')
dataset_control['vcomp'].mean('lon')[0,:,:].plot.contourf(cmap = 'bwr',yincrease=False,cbar_kwargs={'label': 'Northward wind speed (m/s)'})
plt.ylabel('Pressure (hPa)')
plt.xlabel('Latitude')
plt.title('Meridional Wind (Earth)')

plt.figure('Meridional Wind (Experiment)')
dataset_exp['vcomp'].mean('lon')[0,:,:].plot.contourf(cmap = 'bwr',yincrease=False,cbar_kwargs={'label': 'Northward wind speed (m/s)'})
plt.ylabel('Pressure (hPa)')
plt.xlabel('Latitude')
plt.title('Meridional Wind (Experiment)')

vcomp_diff = dataset_control['vcomp'].values - dataset_exp['vcomp'].values
lat =  dataset_control['lat'].values
pfull =  dataset_control['pfull'].values

plt.figure('Meridional Wind Difference (Earth minus Experiment)')
plt.contourf(lat,pfull,np.mean(vcomp_diff,axis=3)[0,:,:],cmap='bwr')
plt.gca().invert_yaxis()
plt.colorbar(label='Northward wind speed (m/s)')
plt.title('Meridional Wind Difference (Earth minus Experiment)')
plt.ylabel('Pressure (hPa)')
plt.xlabel('Latitude')


plt.figure('Surface Temperature (Earth)')
dataset_control['t_surf'][0,:,:].plot.contourf(cmap = 'hot',cbar_kwargs={'label': '(K)'})
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.title('Surface Temperature (Earth)')

plt.figure('Surface Temperature (Experiment)')
dataset_exp['t_surf'][0,:,:].plot.contourf(cmap = 'hot',cbar_kwargs={'label': '(K)'})
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.title('Surface Temperature (Experiment)')

t_surf_diff = dataset_control['t_surf'].values - dataset_exp['t_surf'].values
lat =  dataset_control['lat'].values
lon =  dataset_control['lon'].values


plt.figure('Surface Temperature Difference (Earth minus Experiment)')
plt.contourf(lon,lat,t_surf_diff[0,:,:],cmap='bwr')
plt.ylabel('Latitude')
plt.xlabel('Longitude')

plt.colorbar(label='(K)')
plt.title('Surface Temperature Difference (Earth minus Experiment)')
#plt.ylabel('Pressure (hPa)')
#plt.xlabel('Latitude')

plt.show()

