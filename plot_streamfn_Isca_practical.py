'''calculate and plot streamfunction for Isca experiments
'''

import numpy as np
import xarray as xr
import os, sys

#import calculate_PV as cPV
import colorcet as cc
import string

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import (cm, colors,cycler)
import matplotlib.path as mpath
import matplotlib

import pandas as pd

class nf(float):
    def __repr__(self):
        s = f'{self:.1f}'
        return f'{self:.0f}' if s[-1] == '0' else s

def make_colourmap(vmin, vmax, step, **kwargs):
    '''
    Makes a colormap from ``vmin`` (inclusive) to ``vmax`` (exclusive) with
    boundaries incremented by ``step``. Optionally includes choice of color and
    to extend the colormap.
    '''
    col = kwargs.pop('col', 'viridis')
    extend = kwargs.pop('extend', 'both')

    boundaries = list(np.arange(vmin, vmax, step))

    if extend == 'both':
        cmap_new = cm.get_cmap(col, len(boundaries) + 1)
        colours = list(cmap_new(np.arange(len(boundaries) + 1)))
        cmap = colors.ListedColormap(colours[1:-1],"")
        cmap.set_over(colours[-1])
        cmap.set_under(colours[0])

    elif extend == 'max':
        cmap_new = cm.get_cmap(col, len(boundaries))
        colours = list(cmap_new(np.arange(len(boundaries))))
        cmap = colors.ListedColormap(colours[:-1],"")
        cmap.set_over(colours[-1])
    
    elif extend == 'min':
        cmap_new = cm.get_cmap(col, len(boundaries))
        colours = list(cmap_new(np.arange(len(boundaries))))
        cmap = colors.ListedColormap(colours[1:],"")
        cmap.set_under(colours[0])

    norm = colors.BoundaryNorm(boundaries, ncolors = len(boundaries) - 1,
                               clip = False)

    return boundaries, cmap_new, colours, cmap, norm



def calc_streamfn(lats, pfull, v, **kwargs):
    '''
    Calculate meridional streamfunction from zonal mean meridional wind.
    
    Parameters
    ----------

    lats   : array-like, latitudes, units (degrees)
    pfull  : array-like, pressure levels, units (Pa) (increasing pressure)
    v      : array-like, zonal mean meridional wind, dimensions (lat, pfull)
    radius : float, planetary radius, optional, default 3.39e6 m
    g      : float, gravity, optional, default 3.72 m s**-2

    Returns
    -------

    psi   : array-like, meridional streamfunction, dimensions (lat, pfull),
            units (kg/s)
    '''

    radius = kwargs.pop('radius', 3.39e6)
    g      = kwargs.pop('g', 3.72)

    coeff = 2 * np.pi * radius / g

    psi = np.empty_like(v.values)
    for ilat in range(lats.shape[0]):
        psi[0, ilat] = coeff * np.cos(np.deg2rad(lats[ilat]))*v[0, ilat] * pfull[0]
        for ilev in range(pfull.shape[0])[1:]:
            psi[ilev, ilat] = psi[ilev - 1, ilat] + coeff*np.cos(np.deg2rad(lats[ilat])) \
                              * v[ilev, ilat] * (pfull[ilev] - pfull[ilev - 1])
    

    return psi

if __name__ == "__main__":

    # Mars-specific!
    theta0 = 200. # reference temperature
    kappa = 0.25 # ratio of specific heats
    p0 = 610. # reference pressure
    omega = 7.08822e-05 # planetary rotation rate
    g = 3.72076 # gravitational acceleration
    rsphere = 3.3962e6 # mean planetary radius

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

       

    ##### get data #####
    exp = [
        'soc_mars_mk36_per_value70.85_none_mld_2.0',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_lh_rel',
    ]


    fileno = [
        33, 
        33,
    ]

    freq = 'daily'

    p_file = 'atmos_'+freq+'_interp_new_height_temp.nc'


    fig, axs = plt.subplots(1, 3, figsize = (12, 4))

    for i, ax in enumerate(fig.axes):
        ax.tick_params(length = 4, labelsize = 11)
        ax.set_xlim([-90,90])
        ax.set_ylim([1000,0])
        ax.set_xlabel('latitude ($^{\circ}$N)', fontsize = 14)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])
    axs[0].set_ylabel('pressure (hPa)', fontsize = 14)
    
    axs[0].set_title('Control', fontsize = 16, weight = 'bold', y = 1.02)
    axs[1].set_title('Experiment', fontsize = 16, weight = 'bold', y = 1.02)
    axs[2].set_title('Difference', fontsize = 16, weight = 'bold', y = 1.02)

    plt.subplots_adjust(hspace=.2,wspace=.09)


    

    psi_both = []

    for i in range(len(exp)):
        print(exp[i])

        filepath = '$GFDL_DATA/' + exp[i] + '/'
        
        start = 'run00' + str(fileno[i]) + '/'

        i_files = filepath + start + p_file

        d = xr.open_mfdataset(i_files, decode_times = False,
                              concat_dim = 'time', combine='nested')


        ##### reduce dataset #####
        d = d.astype('float32')
        d = d.mean(dim = 'time', skipna = True)
        d = d.sortby('lat', ascending = False)
        d = d.sortby('pfull', ascending = True)
        d = d.mean(dim = 'lon', skipna = True).squeeze()
        d["pfull"] = d.pfull*100
        
        v = d.vcomp
        lat = d.lat
        pfull = d.pfull


        psi = calc_streamfn(lat.load(), pfull.load(), v.load(),
                            radius = rsphere, g = g) / 10 ** 8

        psi_both.append(psi)


    psi_diff = psi_both[0] - psi_both[1]

    vmin = int(np.nanmin(psi_both))
    print(vmin)
    vmax = int(np.nanmax(psi_both))
    print(vmax)
    step = 5

    vmin0 = - int(np.nanmax(abs(psi_diff)))
    vmax0 = - vmin0
    step0 = 2

    d["pfull"] = d.pfull / 100

    ### make colormaps
    boundaries, _, _, cmap, norm = make_colourmap(vmin, vmax, step,
                                        col = 'cet_coolwarm', extend = 'both')

    
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[:2],
                extend='both', ticks=boundaries[slice(None,None,2)],pad=0.18,
                orientation = 'horizontal', aspect = 40)

    cb.set_label(label='$\psi$ ($10^{8}$ kg/s)',
                 fontsize=14)
    cb.ax.tick_params(labelsize=11)

    boundaries0, _, _, cmap0, norm0 = make_colourmap(vmin0, vmax0, step0,
                                        col = 'cet_coolwarm', extend = 'both')

    
    cb0 = fig.colorbar(cm.ScalarMappable(norm=norm0, cmap=cmap0),ax=axs[2],
                extend='both', ticks=boundaries0[slice(None,None,2)],pad=0.18,
                orientation = 'horizontal')

    cb0.set_label(label = '$\psi$ difference ($10^{8}$ kg/s)', fontsize = 14)
    cb0.ax.tick_params(labelsize=11)
    ####
    print()
    axs[0].contourf(lat, pfull, psi_both[0],
                    levels = [-50]+boundaries+[150],norm=norm,cmap=cmap)
    c0=axs[0].contour(lat, pfull, psi_both[0],
                    levels = boundaries[slice(None,None,2)], colors='black',
                    linewidths=0.6)

    c0.levels = [nf(val) for val in c0.levels]
    axs[0].clabel(c0, c0.levels, inline=1, fmt=fmt, fontsize=10)

    axs[1].contourf(lat, pfull, psi_both[1],
                    levels = [-50]+boundaries+[150],norm=norm,cmap=cmap)
    c1=axs[1].contour(lat, pfull, psi_both[1],
                    levels = boundaries[slice(None,None,2)], colors='black',
                    linewidths=0.6)

    c1.levels = [nf(val) for val in c1.levels]
    axs[1].clabel(c1, c1.levels, inline=1, fmt=fmt, fontsize=10)

    axs[2].contourf(lat, pfull, psi_diff,
                    levels = [-50]+boundaries0+[150],norm=norm0,cmap=cmap0)
    c2=axs[2].contour(lat, pfull, psi_diff,
                    levels = boundaries0[slice(None,None,2)], colors='black',
                    linewidths=0.6)

    c2.levels = [nf(val) for val in c2.levels]
    axs[2].clabel(c2, c2.levels, inline=1, fmt=fmt, fontsize=10)

    plt.show(block = True)