import numpy as np
import matplotlib.pyplot as plt
import astropy
from astropy import units as u
from astropy.convolution import Tophat2DKernel
from astropy.coordinates import SkyCoord
import os
import scipy.optimize
#from utils import mkdir, get_binc, format_log_axis
import regions
from regions import CircleSkyRegion
#import format_log_axis

import gammapy
from gammapy.makers import MapDatasetMaker, RingBackgroundMaker, SafeMaskMaker
from gammapy.datasets import MapDataset, MapDatasetOnOff
from gammapy.data import DataStore
from gammapy.maps import MapAxis, WcsGeom, Map
from gammapy.estimators import ExcessMapEstimator

print ('Astropy version:', astropy.__version__)
print ('Gammapy version:', gammapy.__version__)
print ('Regions version:', gammapy.__version__)

import warnings
warnings.filterwarnings('ignore')

W = 10
params = {'figure.figsize': (W, W/(4/3)),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
          'font.size' : 16,
         'text.usetex': True,
         'text.latex.preamble': (            # LaTeX preamble
        r'\usepackage{lmodern}'),
          'font.family': 'lmodern',
          'legend.fontsize': 16,

          }

plt.rcParams.update(params)

work_dir = "/Users/jean/Documents/PhD/gammapy/GRBs/190829A/v17/new_analysis/grb_analysis/"
def save(fig, figname, left = 0.15, bottom = 0.15, right = 0.95, top = 0.95):
    fig.subplots_adjust(left = left, bottom = bottom, top = top, right = right)
    format_fig = ['png','pdf', 'eps']
    for form in format_fig:
        fig.savefig(work_dir + "plots/plots_2D/{}/scripts/grb190829A_{}_{}.{}"
                    .format(args.night,args.night, figname, form))


def get_runlist(run_list):
    cluster1 = [152900, 152901]
    cluster2 = [152902, 152903, 152904]
    cluster3 = [152905, 152906, 152907]

    runs_night1 = [152900, 152901, 152902, 152903, 152904, 152905, 152906, 152907]
    runs_hybrid = [152900, 152901, 152902, 152903, 152904, 152905]
    runs_night2 = [152960, 152961, 152962, 152963, 152965, 152966, 152967, 152968, 152969, 152970]
    runs_night3 = [153040, 153041, 153042, 153043, 153044, 153047, 153048, 153049, 153050] # 153045 (too short)


    options = ["night1","night2","night3","all_nights","cluster1","cluster2","cluster3"]

    if run_list not in options:
        print("Invalid option,use\n")
        print(options)
        return []

    else:
        if run_list == 'night1':
            runs = runs_night1
        elif run_list == 'night2':
            runs = runs_night2
        elif run_list == 'night3':
            runs = runs_night3
        elif run_list == 'all_nights':
            runs = runs_night1 + runs_night2 + runs_night3
        elif run_list == "cluster1":
            runs = cluster1
        elif run_list == "cluster2":
            runs = cluster2
        elif run_list == "cluster3":
            runs = cluster3
        elif run_list == "hybrid":
            runs = runs_hybrid
        return runs


def load_data(runs):
    # Load FITS data from H.E.S.S database/local cpu
    ds = DataStore.from_dir('$GAMMAPY_DATA/std_ImPACT_fullEnclosure')
    observations = ds.get_observations(runs)

    return observations, ds

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description = "Perform fit to data stacked or joint")
    parser.add_argument('--night', dest = "night", type = str, required = True, help = 'Select a night, options are ["night1","night2","night3","all_nights","cluster1","cluster2","cluster3","hybrid"]')

    args = parser.parse_args()

    name_file = 'ringBkg_'+args.night
    print(name_file)


    # we load the data here:
    runs = get_runlist(args.night)
    observations, datastore = load_data(runs)


    ra = 44.544
    dec = -8.958

    source_pos = SkyCoord(ra, dec, frame = 'icrs', unit = 'deg')


    energy_axis = MapAxis.from_edges (
    np.logspace(-0.875, 1.75, 43), unit = 'TeV', name = 'energy', interp = 'log'

                                 )

    geom = WcsGeom.create(
        skydir=source_pos,
        binsz = 0.02,
        width = (5, 5),
        frame = 'icrs',
        axes = [energy_axis],
        )




    geom_image = geom.to_image().to_cube([energy_axis.squash()])


    maker = MapDatasetMaker( selection = ['counts', 'exposure', 'background'])

    maker_safe_mask = SafeMaskMaker(methods = ['offset-max'], offset_max = 2.5 *u.deg )

    reference = MapDataset.create(geom)

    datasets = []

    for obs in observations:
        dataset = maker.run(reference, obs)
        dataset = maker_safe_mask.run(dataset, obs)
        datasets.append(dataset)

    regions = CircleSkyRegion(center = source_pos, radius = 0.3*u.deg)

    mask = Map.from_geom(geom)

    mask.data = mask.geom.region_mask([regions], inside = False)


    ring_maker = RingBackgroundMaker (
    r_in = 0.5*u.deg, width = 0.3 *u.deg, exclusion_mask = mask
    )

    # for this we need to stack all observations together.
    stacked_on_off = MapDatasetOnOff.create(geom = geom_image, name = 'grb-stacked')
    for dataset in datasets:
        dataset_image = dataset.to_image()
        dataset_on_off = ring_maker.run(dataset_image)
        stacked_on_off.stack(dataset_on_off)

    print (stacked_on_off)


    fig = plt.figure()
    f_exp_on = stacked_on_off.exposure.sum_over_axes().plot(add_cbar = True)[0]
    save(fig, '2D-On-exposure-map_low_level')


    # Using a convolution radius of 0.04 degrees
    estimator = ExcessMapEstimator(0.05 * u.deg, )
    lima_maps = estimator.run(stacked_on_off, steps="ts")


    fig = plt.figure()
    fig,ax,_ = stacked_on_off.counts.sum_over_axes().plot(add_cbar = True)
    save(fig, '2D-On-counts_maps_low_level')


    fig = plt.figure()
    fig, ax,_ = stacked_on_off.counts_off.sum_over_axes().plot(add_cbar = True)
    save(fig, '2D-off-counts-maps_low_level')

    map_excess_clipped = lima_maps['excess'].copy()
    map_excess_clipped.data = map_excess_clipped.data.clip(min = 0)

    fig = plt.figure()
    fig, ax,_ = map_excess_clipped.sum_over_axes().plot(add_cbar = True)
    save(fig, '2D-excess-map_low_level')


    map_excess_excl = lima_maps['excess'].copy()
    map_excess_excl.data[:] *= mask.data[0]
    maxentry = np.abs(map_excess_excl.data).max()

    fig = plt.figure()
    fig,ax,_ = map_excess_excl.sum_over_axes().plot(
        add_cbar = True, cmap='coolwarm', vmin = -maxentry, vmax=maxentry
                                                   )
    save(fig, '2D-excess_withexclusion_low_level')


    map_sign_clipped = lima_maps['significance'].copy()
    map_sign_clipped.data = map_sign_clipped.data.clip(min=0) # do not clip I changed this for night2 and 3

    fig = plt.figure()
    fig,ax,_ = map_sign_clipped.sum_over_axes().plot(add_cbar = True)
    save(fig, '2D-significance-map_low_level')


    map_sign_excl = lima_maps['significance'].copy()

    # map_sign_excl.data *= mask.data[0]

    fig = plt.figure()
    fig,ax,cb = map_sign_excl.sum_over_axes().plot(
        add_cbar = True, cmap = 'coolwarm', vmin=-5, vmax=5,
                                                  )
    save(fig, '2D-significance-map_with_exclusion_low_level')

    sdbins = np.linspace (-5, 8,131)
    def get_binc(bins):
        bin_center = (bins[:-1] + bins[1:]) / 2
        return bin_center
    reg_inner = CircleSkyRegion(source_pos, 2.25 * u.deg)
    inner = geom.region_mask([reg_inner], inside=True)[0]

    sign_inner = lima_maps['significance'].sum_over_axes().data[inner]

    sign_excl = map_sign_excl.sum_over_axes().data.copy()
    sign_excl[~mask.data[0].astype(bool)] = -999.
    sign_excl_inner = sign_excl[inner]
    sign_inner[np.isnan(sign_excl_inner)] = -999.
    sign_excl_inner[np.isnan(sign_excl_inner)] = -999.

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.12, 0.12, 0.85, 0.8])
    ax.grid(ls='--')
    ax.set_yscale('log')
    ax.set_xlabel('Significance', fontsize =20)
    ax.set_ylabel('Entries', fontsize =20)
    h = ax.hist(sign_inner, bins=sdbins, histtype='step', color='k', lw=2, zorder=5, label = 'all bins')[0]
    h = ax.hist(sign_excl_inner, bins=sdbins, histtype='step', color='gray', lw=2, zorder=3, label = 'off bins')[0]

    gaus = lambda x,amp,mean,sigma:amp*np.exp(-(x-mean)**2/2/sigma**2)
    xv = np.linspace(sdbins[0], sdbins[-1], 1000)
    res = scipy.optimize.curve_fit(gaus, get_binc(sdbins), h, p0=[h.max(), 0., 1.])
    pars = res[0]
    errs = np.sqrt(np.diag(res[1]))
    ax.plot(xv, gaus(xv, pars[0], pars[1], pars[2]), color='tab:red', lw=2, zorder=7)

    ax.text(0.98, 0.96, 'Mean: ${:.3f}\,\pm\,{:.3f}$\nWidth: ${:.3f}\,\pm\,{:.3f}$'.format(pars[1],errs[1], pars[2], errs[2]),
            ha='right', va='top', bbox=dict(edgecolor='tab:red', facecolor='white'), transform=ax.transAxes, fontsize=16)

    ax.plot(xv, gaus(xv, h.max(), 0, 1), color='tab:blue', lw=2, zorder=6)

    ax.text(0.98, 0.81, 'Mean: $0$\nWidth: $1$', ha='right', va='top',fontsize=16,
            bbox=dict(edgecolor='tab:blue', facecolor='white'), transform=ax.transAxes)

    ax.set_xlim(sdbins[0], sdbins[-1])
    ax.set_ylim(bottom=0.3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(loc = 'upper left',fontsize=16)
    # format_log_axis(ax.yaxis)
    save(fig, '1D_significance_distribution_low_level')


    # Theta^2 plot
    tsbins = np.linspace(0, 0.3, 121)
    excess_noconv = stacked_on_off.excess
    excess_noconv_err = np.sqrt(stacked_on_off.counts.sum_over_axes().data +\
                                stacked_on_off.alpha.data**2 * stacked_on_off.counts_off.sum_over_axes().data)
    thsq = stacked_on_off.counts.geom.get_coord().skycoord.separation(source_pos).value**2
    thsq_excess = np.histogram(thsq, bins=tsbins, weights=excess_noconv)[0]
    thsq_excess_err = np.sqrt(np.histogram(thsq, bins=tsbins, weights=excess_noconv_err**2)[0])

    tsbinc = get_binc(tsbins)
    xerr = [tsbinc - tsbins[:-1], tsbins[1:] - tsbinc]

    # create figure
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_axes([0.12, 0.15, 0.85, 0.82])
    ax.grid(ls='--')
    ax.set_xlabel(r'$\theta^2$ [$\mathrm{deg}^2$]', fontsize = 20)
    ax.set_ylabel('Excess entries', fontsize = 20)

    # plot distributions
    pexc = ax.errorbar(tsbinc, thsq_excess, xerr=xerr, yerr=thsq_excess_err, linestyle='None',
                       ecolor='tab:red', capsize=0, elinewidth=2)

    # set axis limits
    ax.set_xlim(tsbins[0], 0.15)
    #ax.set_ylim(-15, 50)
    ax.tick_params(axis='both', which='major', labelsize=20)
    save(fig, '1D-thetasquare_distribution_low_level')


plt.show()
