import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from pathlib import Path
import scipy.optimize

from gammapy.data import DataStore
from gammapy.maps import MapAxis, WcsGeom
from gammapy.maps import WcsGeom, MapAxis, Map
from gammapy.datasets import MapDataset, Datasets
from gammapy.makers import MapDatasetMaker, SafeMaskMaker, FoVBackgroundMaker
from gammapy.modeling.models import (
    PowerLawSpectralModel, 
    SkyModel, 
    PointSpatialModel, 
    Models,
)
from gammapy.modeling import Fit
from gammapy.estimators import ExcessMapEstimator
from gammapy.estimators.utils import find_peaks
from gammapy.visualization import plot_contour_line
from gammapy.estimators import FluxPointsEstimator
from gammapy.datasets import (
    Datasets,
    SpectrumDataset,
    SpectrumDatasetOnOff,
    FluxPointsDataset,
)

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion

from gammapy.data import DataStore
from gammapy.datasets import MapDataset, FluxPointsDataset
from gammapy.maps import WcsGeom, MapAxis, Map
from gammapy.makers import MapDatasetMaker, SafeMaskMaker, FoVBackgroundMaker
from gammapy.modeling.models import (
    SkyModel,
    PowerLawSpectralModel,
    PointSpatialModel,
)
from gammapy.modeling import Fit
from gammapy.estimators import FluxPointsEstimator


work_dir = "/Users/jean/Documents/PhD/gammapy/GRBs/190829A/v17/new_analysis/grb_analysis/"
def save(fig, figname, left = 0.15, bottom = 0.15, right = 0.95, top = 0.95):
    fig.subplots_adjust(left = left, bottom = bottom, top = top, right = right)
    format_fig = ['png','pdf'] # 'eps'
    for form in format_fig:
        fig.savefig(work_dir + "plots/plots_3D/{}/scripts/grb190829A_{}_{}.{}"
                    .format(args.night,args.night, figname, form))
W = 14

params = {'figure.figsize': (W, W/(4/3)),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
          'font.size' : 14,
         'text.usetex': True,
          'font.family': "sans-serif",
          'legend.fontsize': 14,
          }
plt.rcParams.update(params)

def get_runlist(run_list):
    cluster1 = [152900, 152901]
    cluster2 = [152902, 152903, 152904]
    cluster3 = [152905, 152906, 152907]

    runs_night1 = [152900, 152901, 152902, 152903, 152904, 152905, 152906, 152907]
    runs_hybrid = [152900, 152901, 152902, 152903, 152904, 152905]
    runs_night2 = [152960, 152961, 152962, 152963, 152965, 152966, 152967, 152968, 152969, 152970]
    runs_night3 = [153040, 153041, 153042, 153043, 153044, 153047, 153048, 153049, 153050] # 153045 (too short)


    options = ["night1","night2","night3","all","cluster1","cluster2","cluster3"]

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
        elif run_list == 'all':
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

    name_file = '3D-Bkg_'+args.night

    print(name_file)


    # we load the data here:
    runs = get_runlist(args.night)
    observations, datastore = load_data(runs)
    #print(observations)


    position = SkyCoord(44.544,-8.958, unit='deg', frame='icrs')
    #energy = MapAxis.from_energy_bounds(0.1,40,nbin=16, per_decade=True, unit='TeV')
    #energy_true = MapAxis.from_energy_bounds(0.05,500,nbin=16, per_decade=True, unit='TeV', name="energy_true")
    energy_axis = MapAxis.from_edges(
        np.logspace(-1, 2, 49), unit='TeV', name='energy', interp='log'
        )

    geom = WcsGeom.create(    
        skydir=position,
        binsz=0.02,
        width=(4, 4),
        frame="icrs",
        proj="CAR",
        axes=[energy_axis],
        )

    stacked = MapDataset.create(
        geom=geom, name="grb_stacked"
        )

    offset_max = 2.5 * u.deg
    maker = MapDatasetMaker()
    maker_safe_mask = SafeMaskMaker(methods=['offset-max', 'edisp-bias'], offset_max=2.5*u.deg, bias_percent=10)

    circle = CircleSkyRegion(
        center=position, radius=0.3 * u.deg
        )

    # Exclusion region
    exclusion_ra = 44.106
    exclusion_dec = -8.98981
    exclusion_radius = 0.2
    exclusion_region = CircleSkyRegion(
        center= SkyCoord(exclusion_ra, exclusion_dec, unit="deg", frame="icrs"),
        radius= exclusion_radius * u.deg,
        )

    skydir = position.icrs
    exclusion_mask = Map.create(
        npix=(150, 150), binsz=0.05, skydir=skydir, proj="TAN", frame="icrs")
    data = geom.region_mask(regions=[circle, exclusion_region], inside=False)
    exclusion_mask = Map.from_geom(geom=geom, data=data)
    maker_fov = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)





    datasets = []
    for obs in observations:
        #First a cutout of the target map is produced
        cutout = stacked.cutout(
            obs.pointing_radec, width=2 * offset_max, name=f"obs-{obs.obs_id}",
            mode="partial"
            )
        # A MapDataset is filled in this cutout geometry
        dataset = maker.run(cutout, obs)
        # The data quality cut is applied
        dataset = maker_safe_mask.run(dataset, obs)

        # fit background model
        dataset = maker_fov.run(dataset)

        print(
            f"Background norm obs {obs.obs_id}: {dataset.background_model.norm.value:.2f}"
            )

        # if the background norm is completely off scale don't stack run

        if np.abs(dataset.background_model.norm.value-1.)>0.5:
            print("Dropping run.")
            continue

        # The resulting dataset cutout is stacked onto the final one
        datasets.append(dataset)
        stacked.stack(dataset)

    print(stacked)

    # Counts map
    fig = plt.figure()
    stacked.counts.sum_over_axes().plot(add_cbar=True);
    save(fig, '3D-2D-Stacked_Counts_maps_analysis')


    # Background map
    fig = plt.figure()
    stacked.background_model.evaluate().sum_over_axes().plot(add_cbar= True);
    save (fig, '3d_2d-Background_maps_analysis_stacked')


    fig = plt.figure(figsize=(14,6))
    ax1=plt.subplot(121, projection=stacked.counts.geom.wcs)
    _ = stacked.counts.sum_over_axes().smooth(0.05 * u.deg).plot(ax=ax1,  add_cbar=True)
    ax1.set_xlabel('Right Ascession', fontsize=20)
    ax1.set_ylabel('Declination', fontsize=20)
    ax2=plt.subplot(122, projection=stacked.counts.geom.wcs)
    _ = stacked.residuals().sum_over_axes().smooth(0.05 * u.deg).plot(ax=ax2, add_cbar=True)
    ax2.set_xlabel('Right Ascession', fontsize=20)
    ax2.set_ylabel('Declination', fontsize=20)
    save(fig, '3d_2D-counts_and_residial_3dmaps_analysis')



    image_dataset = stacked.to_image(name="grb-image")
    excess_estimator = ExcessMapEstimator(correlation_radius='0.05 deg')
    excess_maps = excess_estimator.run(image_dataset, steps="all")



    fig = plt.figure()
    background_map = excess_maps['background'].sum_over_axes().plot(add_cbar=True);
    save(fig, '3d_to_2Dimage_background_maps')


    fig = plt.figure()
    flux_map = excess_maps["ul"].sum_over_axes().plot(add_cbar=True);
    save(fig, '3d_to_2Dimage_upperlimits_map')



    fig = plt.figure()
    TS_map = excess_maps["ts"].sum_over_axes().plot(add_cbar=True)
    save(fig, '3d_to_2Dimage_TS_map')


    fig = plt.figure()
    map_excess_clipped = excess_maps['excess'].copy()
    map_excess_clipped.data = map_excess_clipped.data.clip(min = 0)
    fig, ax,_ = map_excess_clipped.sum_over_axes().plot(add_cbar = True)
    save(fig, '3d_to_2Dimage_excess_map')



    def plot_sqrt_ts_map(sqrt_ts, add_cbar=True, bins=100, figsize=(10,7)):
        from matplotlib.gridspec import GridSpec
        from scipy.stats import norm
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(nrows=4, ncols=3, figure=fig)
        ax_map = fig.add_subplot(gs[:,0:2], projection=sqrt_ts.geom.wcs)
        ax_hist = fig.add_subplot(gs[1:3,2])
        ax_res = fig.add_subplot(gs[3,2])

        sqrt_ts.plot(ax=ax_map, add_cbar=add_cbar)

        # plot significance histogram

        significance = sqrt_ts.data.flatten()
        vals = ax_hist.hist(significance, bins=bins, log=True, density=True)

        # Fit the significance distribution

        valid = np.where(np.abs(significance)<4.0)
        mu, std = norm.fit(significance[valid])
        x = np.linspace(-5, 5, 50)
        p = norm.pdf(x, mu, std)
        ax_hist.plot(x, p, lw=2, color="black")

        plt.setp(ax_res, frame_on=False, xticks=(), yticks=())
        ax_res.text(0.3,0.7, r'$\mu$ = %.3f' %mu)
        ax_res.text(0.3,0.3, r'$\sigma$ = %.3f' %std)
        save(fig,'3d_to_2Dimage_Sign_map_hist_dist')

    tsmap = plot_sqrt_ts_map(excess_maps["significance"].sum_over_axes())

    # Residual significance map
    fig = plt.figure()
    res_sign = excess_maps['significance'].sum_over_axes().plot(add_cbar=True, cmap='coolwarm', vmin=-5, vmax=5);
    save(fig, '3d-2D-residual_significance_map_3danalysis')


    sources = find_peaks(excess_maps["significance"].get_image_by_idx((0,)), threshold= 3)
    candidate_position = SkyCoord(sources['ra'], sources["dec"], unit="deg", frame="icrs")


    # Plot sources on top of significance sky image
    fig = plt.figure()
    excess_maps["significance"].sum_over_axes().plot(add_cbar=True);
    plt.ylabel('Declination', fontsize = 16)
    plt.xlabel("Right Ascension", fontsize = 16)
    plt.gca().scatter(
        candidate_position.ra.deg,
        candidate_position.dec.deg,
        transform=plt.gca().get_transform("icrs"),
        color="none",
        edgecolor="white",
        marker="o",
        s=200,
        lw=3,
        );

    save (fig, 'peak_significance_skyimage_source_detection')


    spatial_model = PointSpatialModel(
        lon_0=candidate_position[0].ra,
        lat_0=candidate_position[0].dec,
        frame="icrs")

    spectral_model = PowerLawSpectralModel(
        index=2.501,
        amplitude=1.e-11 * u.Unit("1 / (cm2 s TeV)"),
        reference=0.55 * u.TeV,
        )

    sky_model = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name="grb"
        )

    # We freeze the spectral index 
    spectral_model.index.frozen = True

    image_dataset.models.append(sky_model)

    print(image_dataset.models)

    fit_2d = Fit([image_dataset])
    fit_2d_result = fit_2d.run()
    print(fit_2d_result)

    print(fit_2d_result.parameters.to_table())


    def plot_sqrt_ts_map(sqrt_ts, add_cbar=True, bins=100, figsize=(10,7)):
        from matplotlib.gridspec import GridSpec
        from scipy.stats import norm
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(nrows=4, ncols=3, figure=fig)
        ax_map = fig.add_subplot(gs[:,0:2], projection=sqrt_ts.geom.wcs)
        ax_hist = fig.add_subplot(gs[1:3,2])
        ax_res = fig.add_subplot(gs[3,2])

        
        sqrt_ts.plot(ax=ax_map, add_cbar=add_cbar)
        
        # plot significance histogram
        significance = sqrt_ts.data.flatten()
        vals = ax_hist.hist(significance, bins=bins, log=True, density=True)

        # Fit the significance distribution
        valid = np.where(np.abs(significance)<4.0)
        mu, std = norm.fit(significance[valid])
        x = np.linspace(-5, 5, 50)
        p = norm.pdf(x, mu, std)
        ax_hist.plot(x, p, lw=2, color="black")

        plt.setp(ax_res, frame_on=False, xticks=(), yticks=())
        ax_res.text(0.3,0.7, r'$\mu$ = %.3f' %mu)
        ax_res.text(0.3,0.3, r'$\sigma$ = %.3f' %std)
        save(fig,'3d_sign_dist_2d_fit')


    excess_maps_2 = excess_estimator.run(image_dataset)
    plot_sqrt_ts_map(excess_maps_2["significance"].sum_over_axes());

    sdbins = np.linspace(-5, 8, 131)

    def get_binc(bins):
        bin_center = (bins[:-1] + bins[1:])/2
        return bin_center
    reg_inner = CircleSkyRegion(position, 2.25*u.deg)
    inner = geom.region_mask([reg_inner], inside=True)[0]

    sign_inner = excess_maps['significance'].sum_over_axes().data[inner];

    # Significance map with source excluded

    map_sign_excl = excess_maps['significance'].copy()
    sign_excl = map_sign_excl.sum_over_axes().data.copy()
    sign_excl[~exclusion_mask.data[0].astype(bool)] = -999.
    sign_excl_inner = sign_excl[inner]
    sign_inner[np.isnan(sign_inner)] = -999.
    sign_excl_inner[np.isnan(sign_excl_inner)] = -999.



    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.12, 0.12, 0.85, 0.8])
    ax.grid(ls='--')
    ax.set_yscale('log')
    ax.set_xlabel('Significance')
    ax.set_ylabel('Entries')
    h = ax.hist(sign_inner, bins=sdbins, histtype='step', color='k', lw=2, label = 'all bins', zorder=5)[0]
    h = ax.hist(sign_excl_inner, bins=sdbins, histtype='step', color='gray', lw=2, label ='off bins',zorder=3)[0]

    gaus = lambda x,amp,mean,sigma:amp*np.exp(-(x-mean)**2/2/sigma**2)
    xv = np.linspace(sdbins[0], sdbins[-1], 1000)
    res = scipy.optimize.curve_fit(gaus, get_binc(sdbins), h, p0=[h.max(), 0., 1.])
    pars = res[0]
    errs = np.sqrt(np.diag(res[1]))
    ax.plot(xv, gaus(xv, pars[0], pars[1], pars[2]), color='tab:red', lw=2, zorder=7)

    ax.text(0.98, 0.96, 'Mean: ${:.3f}\,\pm\,{:.3f}$\nWidth: ${:.3f}\,\pm\,{:.3f}$'.format(pars[1], errs[1], pars[2], errs[2]),
            ha='right', va='top', bbox=dict(edgecolor='tab:red', facecolor='white'), transform=ax.transAxes)

    ax.plot(xv, gaus(xv, h.max(), 0, 1), color='tab:blue', lw=2, zorder=6)

    ax.text(0.98, 0.81, 'Mean: $0$\nWidth: $1$', ha='right', va='top',
            bbox=dict(edgecolor='tab:blue', facecolor='white'), transform=ax.transAxes)

    ax.set_xlim(sdbins[0], sdbins[-1])
    ax.set_ylim(bottom=0.3)
    ax.legend(loc = 'upper left', fontsize = 20)
    #format_log_axis(ax.yaxis)
    save(fig, "3d_2D_fitting_Sign_distribution")



    conf_lon = fit_2d.confidence(spatial_model.lon_0)
    conf_lat = fit_2d.confidence(spatial_model.lat_0)

    contours_1 = fit_2d.minos_contour(spatial_model.lon_0, spatial_model.lat_0, numpoints=20 )
    contours_2 = fit_2d.minos_contour(spatial_model.lon_0, spatial_model.lat_0, numpoints=20, sigma=2)


    fig = plt.figure()
    ax=plt.subplot()
    plot_contour_line(ax=ax, x=contours_1['x'], y=contours_1['y'],color = 'blue')
    plot_contour_line(ax=ax, x=contours_2['x'], y=contours_2['y'], color = 'darkblue')
    countour = plt.errorbar(x=[spatial_model.lon_0.value],
                 y=[spatial_model.lat_0.value], 
                 xerr=[[conf_lon["errn"]],[conf_lon["errp"]]],
                 yerr=[[conf_lat["errn"]],[conf_lat["errp"]]],
                 fmt='o'
                )

    plt.xlabel('Right Ascension', fontsize= 20)
    plt.ylabel('Declination', fontsize= 20)
    save(fig, '3d_2d_fit_Source_position_contours')





plt.show()


















