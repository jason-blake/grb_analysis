import matplotlib.pyplot as plt
# Check package versions
import gammapy
import numpy as np
import astropy
import regions

print("gammapy:", gammapy.__version__)
print("numpy:", np.__version__)
print("astropy", astropy.__version__)
print("regions", regions.__version__)

from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from gammapy.maps import Map, MapAxis
from gammapy.modeling import Fit
from gammapy.data import DataStore
from gammapy.datasets import (
    Datasets,
    SpectrumDataset,
    SpectrumDatasetOnOff,
    FluxPointsDataset,
)
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    create_crab_spectral_model,
    SkyModel,
    AbsorbedSpectralModel,
    Absorption,
)
from gammapy.makers import (
    SafeMaskMaker,
    SpectrumDatasetMaker,
    ReflectedRegionsBackgroundMaker,
)

from gammapy.estimators import FluxPointsEstimator
from gammapy.visualization import plot_spectrum_datasets_off_regions


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

work_dir = "/Users/jean/Documents/PhD/gammapy/GRBs/190829A/v17/new_analysis/grb_analysis/_scripts/"
pathdata = '$GAMMAPY_DATA/std_ImPACT_fullEnclosure'
def load_data(runs):
    # Load FITS data from H.E.S.S database/local cpu
    ds = DataStore.from_dir(pathdata)
    observations = ds.get_observations(runs)

    return observations, ds



def save(fig, figname, left = 0.15, bottom = 0.15, right = 0.95, top = 0.95):
    fig.subplots_adjust(left = left, bottom = bottom, top = top, right = right)
    format_fig = ['png','pdf'] # used also this 'eps', but it's heavy 
    for form in format_fig:
        fig.savefig(work_dir + "plots/plots_1D/{}/scripts/grb190829A_{}_{}.{}"
                    .format(args.night,args.night, figname, form))

### taken from C.Rumori #########

def my_residuals(self, method = "diff"):
    fluxpoints = self.data
    data = fluxpoints.table[fluxpoints.sed_type]

    model = self.flux_pred()

    residuals = self._compute_residuals(data, model, method)
    # Remove residuals for upper_limits
    residuals[fluxpoints.is_ul] = np.nan

    fluxpoints = self.data

    model = self.flux_pred()
    yerr = fluxpoints._plot_get_flux_err(fluxpoints.sed_type)

    if method == "diff":
        unit = yerr[0].unit
        yerr = yerr[0].to_value(unit), yerr[1].to_value(unit)
    elif method == "diff/model":
        unit = ""
        yerr = (yerr[0] / model).to_value(""), (yerr[1] / model).to_value(unit)
    else:
        raise ValueError("Invalid method, choose between 'diff' and 'diff/model'")

    return residuals, yerr


W = 14

params = {'figure.figsize': (W, W/(4/3)),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
          'font.size' : 16,
         'text.usetex': True,
          'font.family': "sans-serif",
          'legend.fontsize': 16,
          }
plt.rcParams.update(params)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description = "Perform fit to data stacked or joint")
    parser.add_argument('--night', dest = "night", type = str, required = True, help = 'Select a night, options are ["night1","night2","night3","all_nights","cluster1","cluster2","cluster3","hybrid"]')
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

    name_file = 'reflectedBg_'+args.night+'_'+args.eblmodel+'_alpha'+str(args.alpha)+'_'+args.fit_type
    print(name_file)


    # we load the data here:
    runs = get_runlist(args.night)
    observations, datastore = load_data(runs)

    #Configure the target and on/exclusion region(s)
    #here we will add an exclusion region for the star
    # at 0,4 deg away from the GRB just to be on the safe side!


    # Target on region

    ra, dec = 44.544, -8.958 # position of GRB 190829A
    target_position = SkyCoord(ra, dec, unit="deg", frame="icrs")
    on_region_radius = Angle("0.071 deg") # 0.08 tutorial default
    on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)


    # Exclusion region
    exclusion_ra = 44.106
    exclusion_dec = -8.98981
    exclusion_radius = 0.2
    exclusion_region = CircleSkyRegion(
    center= SkyCoord(exclusion_ra, exclusion_dec, unit="deg", frame="icrs"),
    radius= exclusion_radius * u.deg,
    )

    skydir = target_position.icrs
    exclusion_mask = Map.create(
    npix=(150, 150), binsz=0.02, skydir=skydir, proj="TAN", frame="icrs")

    mask = exclusion_mask.geom.region_mask([exclusion_region], inside=False)
    exclusion_mask.data = mask

    ## Start binning in energy!

    e_reco = np.logspace(np.log10(args.ethres, np.log10(40), 40) * u.TeV
    e_true = np.logspace(np.log10(0.05), 2, 200) * u.TeV

    dataset_maker = SpectrumDatasetMaker(
    containment_correction=True, selection=["counts", "aeff", "edisp"]
    )

    bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
    #safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

    safe_mask_masker = SafeMaskMaker(methods=["aeff-default", "edisp-bias"], bias_percent=10)

    dataset_empty = SpectrumDataset.create(
    e_reco=e_reco, e_true=e_true, region=on_region
    )


    datasets = Datasets()

    for obs_id, observation in zip(runs, observations):
        dataset = dataset_maker.run(
            dataset_empty.copy(name=str(obs_id)), observation
        )
        dataset_on_off = bkg_maker.run(dataset, observation)
        dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
        datasets.append(dataset_on_off)

    print(datasets)

    #Print out some source statistics from the reflected bg per run.

    print('Run   Threshold [TeV]\n')
    print('----------------------\n')

    # Show energy threshold for each run
    min_energy = np.inf
    for obs,ds in zip(observations, datasets):
        thres = ds.energy_range[0]
        min_energy = min(min_energy, thres)
        print(obs.obs_id, '{:.4g}'.format(min_energy))

    info_table = datasets.info_table(cumulative = True)
    print("\n Table of source statistics:\n")
    print('------------------------------\n')
    print(info_table)

    print('--------------------------------------------\n')

    print(
    'Excess Counts: '+'{:.6s}'.format(str(info_table['excess'][-1])) + '\n',
    'On Counts: ' +'{:.6s}'.format( str(info_table['n_on'][-1]))+'\n',
    'Off Counts: '+'{:.6s}'.format(str( info_table['n_off'][-1])) + '\n', 
    'Livetime: ' + '{:.6s}'.format(str(info_table['livetime'][-1]/3600)) + '\n',
    'Alpha: ' + '{:.6s}'.format(str(info_table['alpha'][-1])) + '\n',
    'Background: ' + '{:.6s}'.format(str(info_table['background'][-1])) + '\n',
    'Background_rate: '+ '{:.6s}'.format(str(info_table['background_rate'][-1])) +'\n',
    'Excess rate: ' + '{:.6s}'.format(str(info_table['gamma_rate'][-1])) + '\n',
    'Excess/background: ' +'{:.6s}'.format(str(info_table['excess'][-1]/info_table['background'][-1])) + '\n',
    'Significance: ' + '{:.6s}'.format(str(info_table['significance'][-1]))
     )


    # Fit the Spectrum

    #Now build the spectral models
    norm_energy = 0.556

    #These hacks are useful for performing systematic analysis
    #Here alpha is a parameter given by the user.
    #All the reconstructed energies will be shifted by a factor alpha
    #ie E_reco = E_reco*alpha during the spectral fit

    modelpwl = PowerLawSpectralModel(index = 2.0, #these are initialization parameters
                                     amplitude = 1e-11 * u.Unit("cm-2 s-1 TeV-1"),
                                     reference = norm_energy * u.TeV)

    if not ebl:
        print("Doing pure PWL\n")
        print('----------------------------\n')
        spectral_model = modelpwl
    else:
        print("Doing PWL + EBL absoprtion\n")
        print('------------------------------\n')
        absorption = Absorption.read_builtin(args.eblmodel)
        #ebl_abs = AbsorbedSpectralModel.evaluate(, redshift= 0.079)
        # EBL + PWL model
        spectral_model = AbsorbedSpectralModel(
        spectral_model= modelpwl, absorption=absorption, redshift=0.079
        )
        #spectral_model= modelpwl* ebl_abs

        print(spectral_model)

    #Now we create the model with the spectral model we have

    model = SkyModel(spectral_model = spectral_model)


    for dataset in datasets:
        dataset.models = model


    if args.fit_type == "joint":
        print("Performing a joint analysis\n")
        print('------------------------------------')
        fit_joint = Fit(datasets)
        result_joint = fit_joint.run()
        # we make a copy here to compare it later
        model_best_joint = model.copy()

        print("\nJoint fit finished:\n")
        print('--------------------------------------\n')
        print(result_joint)
        result_joint.parameters.to_table().write(work_dir + "../flux_and_fit_results/" + name_file+'_joint_fit_v17.csv', overwrite=True)
        print (result_joint.parameters.to_table())
        print("CSV file with table saved.")

    if args.fit_type == "stacked":
        print("\nPerforming a stacked analysis\n")
        print('--------------------------------------\n')
        dataset_stacked = Datasets(datasets).stack_reduce()
        dataset_stacked.models = model

        stacked_fit = Fit([dataset_stacked])
        result_stacked = stacked_fit.run()

        # make a copy to compare later
        model_best_stacked = model.copy()
        print("\nStacked fit finished:\n")
        print(result_stacked)
        result_stacked.parameters.to_table().write(work_dir + "../flux_and_fit_results/" + name_file+'_stacked_fit_v17.csv', overwrite=True)
        print (result_stacked.parameters.to_table())
        print("CSV file with table saved.")


    # Flix points computation
    # Set binning
    ebounds = np.logspace(np.log10(args.ethres), np.log10(4), args.nbins)
    ebounds = ebounds[ebounds.searchsorted(min_energy.value+1e-4)-1:]

    fpe = FluxPointsEstimator( e_edges = ebounds*u.TeV, reoptimize = True)

    with np.errstate(divide='ignore', invalid='ignore'):
        flux_points = fpe.run(datasets=datasets)

    print(flux_points.table_formatted)
    print('')
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
    for e,f,fl,fh,lim,ul,s in zip(flux_points.table['e_ref'], flux, flux-fluxerr, flux+fluxerr, fluxul, isul, sqrtts):
        if ul:
            print('{:^14.3f}|{:^16s}|{:16s}|{:16s}|{:^16.2f}'.format(e, '< {:.4g}'.format(lim), ' ', ' ', s))
        else:
            print('{:^14.3f}|{:^16.4g}|{:^16.4g}|{:^16.4g}|{:^16.2f}'.format(e, f, f-fl, f+fh, s))

    if args.fit_type == "joint":
        print(result_joint.parameters.names)

    if args.fit_type == "stacked":
        print(result_stacked.parameters.names)


    flux_points_dataset = FluxPointsDataset(data = flux_points, models = model)


    flux_points_dataset.residuals = my_residuals
    residuals , res_err = flux_points_dataset.residuals(flux_points_dataset, method = "diff/model")
    flux_points.table["residuals"] = residuals
    flux_points.table["res_errd"] = res_err[0]
    flux_points.table["res_erru"] = res_err[1]
    flux_points.table['is_ul'] = flux_points.table['ts'] < 2 # set threshold to 2 sigma
    flux_points.write(work_dir + "../flux_and_fit_results/" + name_file+'_flux_points_v17.ecsv',
                      include_names=['e_ref', 'e_min', 'e_max', 'ref_dnde', 'ref_flux', 'ref_eflux', 'ref_e2dnde',
                                    'dnde', 'dnde_err', 'dnde_errp', 'dnde_errn', 'dnde_ul',
                                    'is_ul', 'ts','sqrt_ts',"residuals","res_errd","res_erru"],
                      overwrite=True)


    # Plot results

    fig= plt.figure()
    flux_points.table["is_ul"] = flux_points.table["ts"] < 2
    ax = flux_points.plot(
        energy_power=2, flux_unit="erg-1 cm-2 s-1", color="darkorange")
    flux_points.to_sed_type("e2dnde").plot_ts_profiles(ax=ax);

    ax.grid(ls='--')
    plt.ylabel(r'$\rm{E^2 \, NdNdE (erg/(cm^2 \, s))}$', fontsize = 16)
    plt.xlabel('Energy (TeV)', fontsize=16)
    ax.set_xlim(0.15, 100)
    #ax.set_ylim(5e-15, 1e-10);
    save(fig, name_file+'_ts_profile_spectrum')

    #fig = plt.figure()

    plot_kwargs = {
    "energy_range": [args.ethres, 10] * u.TeV,
    "flux_unit": "erg-1 cm-2 s-1",}

    if args.fit_type == "joint":
        fig= plt.figure()
        flux_points_dataset = FluxPointsDataset(data = flux_points, models = model_best_joint)
        flux_points_dataset.peek();
        plt.grid(ls="--")
        #plt.legend()
        plt.xlim(0.15, 5)
        save(fig, name_file + "_flux_points_peek")
        fig = plt.figure()
        ax = flux_points.plot()
        model_best_joint.spectral_model.plot(**plot_kwargs, ax= ax)
        # plot stacked model
        model_best_joint.spectral_model.plot_error(**plot_kwargs)
        ax.grid(ls="--")
        ax.set_xlim(0.15, 5)
        #ax.set_ylim(5e-15, 3e-10);
        save(fig, name_file+"_spectrum")
    else:
        fig = plt.figure()
        flux_points_dataset = FluxPointsDataset(data = flux_points, models = model_best_stacked)
        flux_points_dataset.peek();
        plt.grid(ls="--")
        plt.xlim(0.15, 5)
        #plt.legend()
        save(fig, name_file + "_flux_points_peek")
        fig= plt.figure()
        ax = flux_points.plot()
        if not ebl:
            model_best_stacked.spectral_model.plot_error(**plot_kwargs)
        model_best_stacked.spectral_model.plot(**plot_kwargs, ax= ax)
        ax.grid(ls="--")
        ax.set_xlim(0.15, 5)
        #ax.set_ylim(5e-15, 3e-10);
        save(fig, name_file+"_spectrum")


plt.show()
