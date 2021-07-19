## General imports
import numpy as np
import matplotlib.pyplot as plt
import astropy
from astropy.convolution import Tophat2DKernel
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy import units as u
import scipy.optimize
from functools import reduce

from pathlib import Path
from regions import CircleSkyRegion
import warnings
warnings.filterwarnings('ignore')

import gammapy
from gammapy.data import DataStore
from gammapy.datasets import (
    MapDataset,
    Datasets,
    FluxPointsDataset
    )
from gammapy.maps import WcsGeom, MapAxis, Map
from gammapy.makers import (
    MapDatasetMaker, 
    SafeMaskMaker, 
    FoVBackgroundMaker
    )
from gammapy.modeling.models import (
    SkyModel,
    PowerLawSpectralModel,
    PointSpatialModel,
    )

from gammapy.modeling import Fit
from gammapy.estimators import FluxPointsEstimator
from gammapy.estimators import ExcessMapEstimator



print('Astropy version:', astropy.__version__)
print('Gammapy version:', gammapy.__version__)


W = 14
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
    format_fig = ['png','pdf'] # used also this 'eps', but it's heavy
    for form in format_fig:
        fig.savefig(work_dir + "plots/plots_1D/{}/scripts/grb190829A_{}_{}.{}"
                    .format(args.night,args.night, figname, form))

def my_residuals(self, method = "diff"):
    fp = self.data
    data = fp.table[fp.sed_type]
    model = self.flux_pred()
    residuals = self._compute_residuals(data, model, method)
    # Remove residuals for upper_limits
    residuals[fp.is_ul] = np.nan
    fp = self.data
    model = self.flux_pred()
    yerr = fp._plot_get_flux_err(fp.sed_type)
    if method == "diff":
        unit = yerr[0].unit
        yerr = yerr[0].to_value(unit), yerr[1].to_value(unit)
    elif method == "diff/model":
        unit = ""
        yerr = (yerr[0] / model).to_value(""), (yerr[1] / model).to_value(unit)
    else:
        raise ValueError("Invalid method, choose between 'diff' and 'diff/model'")
    return residuals, yerr


def get_runlist(run_list):
    #a simple way to return the runlist to analyse
    cluster1 = [152900, 152901]
    cluster2 = [152902, 152903, 152904]
    cluster3 = [152905, 152906, 152907]

    runs_night1 = [152900, 152901, 152902, 152903, 152904, 152905, 152906, 152907]
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
        return runs

def load_data(runs):
    # Load FITS data from H.E.S.S database/local cpu
    ds = DataStore.from_dir("$GAMMAPY_DATA/std_ImPACT_fullEnclosure", 'hdu-index.fits.gz','obs-index.fits.gz',)
    observations = ds.get_observations(runs)

    return observations, ds


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description = "Perform fit to data stacked or joint")
    parser.add_argument('--night', dest = "night", type = str, required = True, help = 'Select a night, options are ["night1","night2","night3","all","cluster1","cluster2","cluster3"]')
    parser.add_argument('--fit_type',dest = "fit_type", required = True, type = str, help = "Options to perform fit are 'stacked' or 'joint'")
    parser.add_argument('--ethres', dest = "ethres", required = True, type = float, help = "energy threshold of the analysis")
    parser.add_argument('--eblmodel', dest = "eblmodel", default ="", type =str, help = "EBL model to use, franceschini, dominguez, finke")
    parser.add_argument("--alpha",dest = "alpha", default = 1.0, type = float, help = "Alpha is a multiplication factor for the energy to perform systematic studies")
    parser.add_argument("--nbins",dest = "nbins", default = 25, type = int, help = "number of bins for the flux points estimation")

    args = parser.parse_args()

    if args.eblmodel == "":
        ebl = False
        args.eblmodel = "noEBL"
    elif args.eblmodel in ["franceschini", "dominguez", "finke"]:
        ebl = True
    else:
        raise AssertionError('Your ebl model is unvalid')

    if args.fit_type not in ["stacked", "joint"]:
        raise AssertionError('Your fit_type model is unvalid')

    name_file = '3d_bkg_'+args.night+'_'+args.eblmodel+'_alpha'+str(args.alpha)+'_'+args.fit_type
    print(name_file)


    #Load the data
    runs = get_runlist(args.night)
    observations, datastore = load_data(runs)


    # Set coordinates of target
    ra_obj = 44.544
    dec_obj = -8.958
    target = SkyCoord(ra_obj, dec_obj, frame='icrs', unit='deg')

    energy_axis = MapAxis.from_edges(
    np.logspace(-1, 2, args.nbins), unit='TeV', name='energy', interp='log'
    )

    geom = WcsGeom.create(
    skydir=target,
    binsz=0.02,
    width=(4, 4),
    frame= 'icrs',
    proj='TAN',
    axes=[energy_axis],
    )

    circle = CircleSkyRegion(
        center=target, radius=0.3 * u.deg
        )
    exclusion_region_star = CircleSkyRegion(
        center=SkyCoord(44.106, -8.9891, unit="deg", frame="icrs"),
        radius=0.2 * u.deg,
        )
    data = geom.region_mask(regions=[circle, exclusion_region_star], inside=False)
    exclusion_mask = Map.from_geom(geom=geom, data=data)
    maker_fov = FoVBackgroundMaker(exclusion_mask=exclusion_mask)

    maker = MapDatasetMaker()
    maker_safe_mask = SafeMaskMaker(methods=['offset-max', 'edisp-bias'], offset_max=2.5*u.deg, bias_percent=10)

    stacked = MapDataset.create(geom)
    datasets = []

    offset_max = 3.5 * u.deg
    for obs in observations:
        # First a cutout of the target map is produced
        cutout = stacked.cutout(
            obs.pointing_radec, width=2 * offset_max, name=f"obs-{obs.obs_id}"
            )
        # A MapDataset is filled in this cutout geometry
        dataset = maker.run(cutout, obs)
        dataset = maker_safe_mask.run(dataset, obs)
        # fit background model
        dataset = maker_fov.run(dataset)
        print(
        f"Background norm obs {obs.obs_id}: {dataset.background_model.norm.value:.2f}"
        )
        datasets.append(dataset)
        stacked.stack(dataset)

    print("\nSummary of the stacked MapDataset ...")
    print('---------------------------------------\n')

    print(stacked)

    fig = plt.figure(figsize=(14,6))
    ax1=plt.subplot(121, projection=stacked.counts.geom.wcs)
    count = stacked.counts.sum_over_axes().smooth(0.05 * u.deg).plot(ax=ax1,  add_cbar=True)
    ax2=plt.subplot(122, projection=stacked.counts.geom.wcs)
    sm = stacked.residuals().sum_over_axes().smooth(0.05 * u.deg).plot(ax=ax2, add_cbar=True);
    save(fig, 'counts_residual_3d_low_level')

    # Counts map
    fig = plt.figure()
    f_counts = stacked.counts.sum_over_axes().plot(add_cbar=True);
    save(fig, 'counts_3d_low_level')

    # Background map
    fig = plt.figure()
    f_bkg = stacked.background_model.evaluate().sum_over_axes().plot(add_cbar=True);
    save(fig, 'background_3d_low_level')

    # Show energy threshold for each run
    print('\nRun   Threshold [TeV]')
    print('----------------------')
    min_energy = np.inf
    for obs,ds in zip(observations, datasets):
        thres = energy_axis.edges[:-1][ds.mask_safe.data[:,100,100]][0]
        min_energy = min(min_energy, thres)
        print(obs.obs_id, '{:.4g}'.format(thres))


    norm_energy = 0.556

    for ds in datasets:
        ds.models.parameters['tilt'].frozen = False

    # Perform fit
    spatial_model = PointSpatialModel(lon_0=target.ra, lat_0=target.dec, frame='icrs')

    spectral_model = PowerLawSpectralModel(
        index     = 2,
        amplitude = 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
        reference = norm_energy * u.TeV,
        )

    modelpwl = spectral_model

    if not ebl:
        print('-----------------------------------')
        print("\n Doing pure PWL")
        spectral_model = modelpwl

    else:
        print("Doing PWL + EBL absoprtion")
        ebl_abs = Absorption.read_builtin(args.eblmodel)
        spectral_model = AbsorbedSpectralModel(modelpwl, ebl_abs, 0.079, "redshift")

    #modelpwl.parameters["index"].frozen = True
    #Now we create the model with the spectral model we have

    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model, name = "grb")

    print(model)


    # Setup fit
    for ds in datasets:
        ds.models.append(model)

    '''
    def plot_spectrum(model, result, label):
        spec = model.spectral_model
        # set covariance on the spectral model
        covar = result.models.parameters.get_subcovariance(spec.parameters)
        spec.parameters.covariance = covar
        energy_range = [0.1, 100] * u.TeV
        spec.plot(energy_range=energy_range, energy_power=2, label=label)
        spec.plot_error(energy_range=energy_range, energy_power=2)

    '''
    if args.fit_type == "joint":
        print("Performing a joint analysis")

        pl_fit = Fit(datasets)
        # Run fit
        with np.errstate(divide='ignore', invalid='ignore'):
            result_joint = pl_fit.run(optimize_opts={"print_level":1})

        #model.spectral_model.parameters.covariance = result_joint.parameters.covariance[2:5,2:5]

        # we make a copy here to compare it later
        model_best_joint = model.copy()
        #model_best_joint.spectral_model.parameters.covariance = (result_joint.parameters.covariance[2:5,2:5])

        print("\nJoint fit finished:")
        print('--------------------------\n')
        print(result_joint)

        result_joint.parameters.to_table().write('../flux_and_fit_results/' + name_file + '_joint_fit_v17.csv', overwrite=True)
        print()
        print (result_joint.parameters.to_table())
        print("CSV file with table saved.")

        '''
        fig = plt.figure()
        ax = plot_spectrum(model_best_joint, result_joint, label="joint")
        plt.legend()
        save (fig, 'joint_model')
        '''
    if args.fit_type == "stacked":
        print("Performing a stacked analysis")

        #dataset_stacked = Datasets(datasets).stack_reduce()
        stacked.models = model
        stacked_fit = Fit([stacked])
        with np.errstate(divide='ignore', invalid='ignore'):
            result_stacked = stacked_fit.run(optimize_opts={"print_level":1})
        # make a copy to compare later
        model_best_stacked = model.copy()
        #model_best_stacked.spectral_model.parameters.covariance = (result_stacked.parameters.covariance[2:5,2:5])

        print("\nStacked fit finished:\n")
        print(result_stacked)
        result_stacked.parameters.to_table().write('../flux_and_fit_results/' + name_file + '_stacked_fit_v17.csv', overwrite=True)
        print (result_stacked.parameters.to_table())
        print("CSV file with table saved.")
        '''
        fig = plt.figure()
        plot_spectrum(model, result_stacked, label="stacked")
        plt.legend()
        '''

    region = CircleSkyRegion(spatial_model.position, radius=0.3*u.deg)
    for obs, ds in zip(observations.ids, datasets):
        #print (obs)
        #print()

        ds.plot_residuals(
            region=region, method="diff/sqrt(model)", vmin=-0.5, vmax=0.5
            )
    
    #plt.show()

    residuals_stacked = Map.from_geom(stacked.counts.geom)
    for dataset in datasets:
        residuals = dataset.residuals()
        residuals_stacked.stack(residuals)
    residuals_stacked = residuals_stacked.slice_by_idx(dict(energy=slice(3,None)))

    #Residual Map
    # (this one likely doesn't take thresholds properly into account)

    residuals_stacked.sum_over_axes().smooth(0.05 * u.deg).plot(cmap = 'coolwarm', add_cbar = True, vmin = -0.25, vmax = 0.25);

    counts_stacked = Map.from_geom(stacked.counts.geom)
    npred_stacked = Map.from_geom(stacked.counts.geom)

    for ds in datasets:
        counts_stacked.data[ds.mask_safe.data] += ds.counts.data[ds.mask_safe.data]
        npred_stacked.data[ds.mask_safe.data] += ds.npred().data[ds.mask_safe.data]

    # Residual excess Map
    # (This one should properly account for different thresholds)
    (counts_stacked - npred_stacked).sum_over_axes().smooth(0.05 * u.deg).plot(add_cbar = True, cmap='coolwarm', vmin = -0.25, vmax = 0.25);



    # compute spectral points
    # set binning
    ebounds = np.logspace(-0.75, np.log10(6), 10)
    ebounds = ebounds[ebounds.searchsorted(min_energy.value+1e-4)-1:]

    # freeze some parameters

    for ds in datasets:
        ds.models.parameters['lon_0'].frozen = True
        ds.models.parameters['lat_0'].frozen = True
        ds.models.parameters['index'].frozen = True
        ds.models.parameters['tilt'].frozen = True

    # Estimate points
    fpe = FluxPointsEstimator(e_edges=ebounds*u.TeV, source = 'grb', reoptimize=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        flux_points = fpe.run(datasets)

    flux_points_dataset = FluxPointsDataset(data = flux_points, models = model)
    flux_points_dataset.residuals = my_residuals

    residuals , res_err = flux_points_dataset.residuals(flux_points_dataset, method = "diff/model")

    flux_points.table["residuals"] = residuals
    flux_points.table["res_errd"] = res_err[0]
    flux_points.table["res_erru"] = res_err[1]

    flux_points.table['is_ul'] = flux_points.table['sqrt_ts'] < 2 # set threshold to 2 sigma

    flux_points.write('../flux_and_fit_results/night1_3d_flux_points_v17_joint_low_level_final.ecsv',
                      include_names=['e_ref', 'e_min', 'e_max',
                                    'dnde', 'dnde_err', 'dnde_errp', 'dnde_errn', 'dnde_ul',
                                    'is_ul', 'sqrt_ts',"residuals","res_errd","res_erru"],
                      overwrite=True)

    # Print and save flux point results
    print('Flux points')
    print('Unit: TeV^-1 cm^-2 s^-1')
    print('')
    print('Energy [TeV]  |      Flux      |    Flux low    |    Flux high   |  Significance')
    flux    = flux_points.table['dnde'].data
    fluxerr = flux_points.table['dnde_err'].data
    fluxul  = flux_points.table['dnde_ul'].data
    sqrtts  = flux_points.table['sqrt_ts'].data
    isul    = sqrtts < 2
    for e,f,fl,fh,lim,ul,s in zip(flux_points.table['e_ref'].data, flux, flux-fluxerr, flux+fluxerr, fluxul, isul, sqrtts):
        if ul:
            print('{:^14.3f}|{:^16s}|{:16s}|{:16s}|{:^16.2f}'.format(e, '< {:.4g}'.format(lim), ' ', ' ', s))
        else:
            print('{:^14.3f}|{:^16.4g}|{:^16.4g}|{:^16.4g}|{:^16.2f}'.format(e, f, f-fl, f+fh, s))


    # Plot results
    flux_points.table['is_ul'] = flux_points.table['sqrt_ts'] < 2 # set threshold to 2 sigma
    f = plt.figure()
    fig = flux_points.plot()
    model.spectral_model.plot_error([min_energy, 100*u.TeV], ax=fig.axes)
    model.spectral_model.plot([min_energy, 100*u.TeV], ax=fig.axes)
    fig.axes.grid(ls='--')
    fig.axes.set_xlim(0.1, 100)
    fig.axes.set_ylim(5e-15, 3e-10);

    f2 = plt.figure()
    fig = flux_points.plot(energy_power=2, flux_unit='TeV-1 cm-2 s-1',color="darkorange")
    flux_points.to_sed_type('e2dnde').plot_ts_profiles(y_unit='TeV cm-2 s-1')
    fig.axes.set_xlim(0.1, 100);

    flux_points.write('../flux_and_fit_results/fitresults_night1_3d_flux_points_v17_scripted.ecsv',
                  include_names=['e_ref', 'e_min', 'e_max',
                                 'dnde', 'dnde_err', 'dnde_errp', 'dnde_errn', 'dnde_ul',
                                 'is_ul', 'sqrt_ts'],
                  overwrite=True)


    result_joint.parameters.to_table().write('../flux_and_fit_results/fitresults_night1_3d_fit17_scripted.csv', overwrite=True)


    # Plot results


    plt.figure(figsize=(8, 5))
    flux_points.table["is_ul"] = flux_points.table["ts"] < 2
    ax = flux_points.plot(
    energy_power=2, flux_unit="erg-1 cm-2 s-1", color="darkorange")
    flux_points.to_sed_type("e2dnde").plot_ts_profiles(ax=ax)
    ax.grid(ls='--')

    plt.ylabel(r'$\rm{E^2 \, NdNdE (erg/(cm^2 \, s))}$', fontsize=16)
    plt.xlabel('Energy (TeV)', fontsize=16)
    ax.set_xlim(0.1, 100)
    #plt.savefig(name_file +'_spectrum_ts_profile.png')



    plt.figure(figsize=(12,9))

    plot_kwargs = {
    "energy_range": [args.ethres, 100] * u.TeV,
    "flux_unit": "erg-1 cm-2 s-1",}
    if args.fit_type == 'joint':
        flux_points_dataset = FluxPointsDataset(data = flux_points, models = model_best_joint)
        flux_points_dataset.peek();
        plt.grid(ls='--')
        #plt.legend()
        #plt.savefig(name_file + '_flux_points_peek.png')

        plt.figure(figsize=(12, 9))
        ax = flux_points.plot()
        model_best_joint.spectral_model.plot(**plot_kwargs, ax= ax)
        # plot stacked model
        model_best_joint.spectral_model.plot_error(**plot_kwargs)
        ax.grid(ls='--')
        #ax.set_xlim(0.1, 100)
        #ax.set_ylim(5e-15, 3e-10);
        #plt.savefig(name_file+'_spectrum.pdf')

    else:

        flux_points_dataset = FluxPointsDataset(data = flux_points, models = model_best_stacked)
        flux_points_dataset.peek();
        plt.grid(ls='--')
        #plt.legend()
        #plt.savefig(name_file + '_flux_points_peek.png')

        plt.figure(figsize=(12, 9))
        ax = flux_points.plot()
        if not ebl:
            model_best_stacked.spectral_model.plot_error(**plot_kwargs)

        model_best_stacked.spectral_model.plot(**plot_kwargs, ax= ax)

        ax.grid(ls='--')
        #ax.set_xlim(0.1, 100)
        #ax.set_ylim(5e-15, 3e-10);
        #plt.savefig(name_file+'_spectrum.pdf')


plt.show()