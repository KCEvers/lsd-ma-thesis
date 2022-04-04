"""
* Download and find appropriate ROIs from the Schaefer2018 atlas based on Long2020's significant dynamic resting-state functional connectivity differences between MDD patients and controls.
* Read and visualise Carhart-Harris2016's data on LSD in healthy subjects (both anatomical and functional, create gif out of functional).
* Apply ROI masks to Carhart-Harris2016 to extract timeseries of voxels corresponding to ROI, and reduce to first two principal components.
* Plot all timeseries (raw and demeaned, standardized, and principal components) - both a summary plot of all ROIs and of each ROI separately per session.
"""

## Downloading data set using anaconda command window:
# Change directory in anaconda command window
# mkdir C:\Users\KyraEvers\Documents\Donders\!Thesis\Data\data_carhart-harris_2016
# cd C:\Users\KyraEvers\Documents\Donders\!Thesis\Data\data_carhart-harris_2016
# datalad clone https://github.com/OpenNeuroDatasets/ds003059.git
# datalad install https://github.com/OpenNeuroDatasets/ds003059.git && cd rawdata && git annex enableremote s3-PUBLIC && datalad get *
# mkdir C:\Users\KyraEvers\Documents\Donders\!Thesis\Data\data_preller_2018
# cd C:\Users\KyraEvers\Documents\Donders\!Thesis\Data\data_preller_2018
# datalad install https://bitbucket.org/katrinpreller/lsd-effects-on-global-brain-connectivity/downloads/subjects.LSD.all.noGSR.udvarsme.surface_gbc_mFz.dtseries.nii
# Link to github (from openneuro): https://github.com/OpenNeuroDatasets/ds003059.git;


# To upgrade pip, open Command Prompt, navigate to
# > cd C:\Users\KyraEvers\PycharmProjects\MA_thesis
# > pip install --upgrade pip

# See nilearn's fetch_neurovault([max_images, …])

# Note: if it is giving errors about not finding a package that was imported using from globalimports import *, make sure you specify the full path, i.e. from python.master.globalimports import *

## Import packages
import vars
from python.master.globalimports import *
from python import processing
from python import visualisation
from python import general
from python import timeseries_analysis

# Parameters
analyze_concat_ts = True # Analyze concatenated timeseries
run_rqa = True
compute_d2 = True


# left off here: check whether rqa measures are computed correctly, excluding NaN

## READ ROI MASKS
# Read txt dataframe with ROI numbers, labels, and coordinates and convert to pandas dataframe
with open(pathlib.Path.joinpath(vars.Schaefer_200_2mm_ROIs_filepath, vars.Schaefer_ROIs_df_filename), 'r') as file:
    Schaefer_ROIs_df = pd.read_csv(io.StringIO(file.read()), sep='\t')

# Create dictionary with ROI labels as keys and ROI masks in numpy format as values
masks = {}
for i, label in enumerate(Schaefer_ROIs_df["ROI_label"]):
    masks[label] = nib.load(pathlib.Path.joinpath(vars.Schaefer_200_2mm_ROIs_filepath,
                                                  "{}_{}.nii".format(vars.prefix_mask,
                                                                     label)).as_posix())  # .get_fdata()

## Specify filenames, paths, and create dictionary of entities to build a file path with.
runs_of_interest = ["01", "03"]  # Specify which runs you want to analyze; these are the scans without music
entities_of_interest = {"run": runs_of_interest,  # Create dictionary specifying which entities you want to extract
                        "datatype": "func",
                        "task": "rest",
                        "suffix": "bold",
                        "extension": "nii.gz"}

# Get BOLD data file names
layout_rawdata = bids.BIDSLayout(vars.path_to_datafolder)  # Create pybids layout
layout_der = bids.BIDSLayout(vars.path_to_derivatives)  # Create pybids layout for derivatives
# subject_nrs = np.array(layout_rawdata.get_subjects(datatype="func")) # Extract subject numbers
rawdata_bold_nifti_files = layout_rawdata.get(
    **entities_of_interest)  # Extract file names of entities of interest from layout

# Get Repetition Time (TR) from random functional image - TR is the same for every one
with open(rawdata_bold_nifti_files[0].get_associations()[0].path, 'r') as file:
    TR = json.load(file)["RepetitionTime"]  # Repetition Time (sec)

# Parameters
method_rqa = 'frr'
thresh_rqa = .05
lmin_rqa = 2
mask_unstand_or_stand="mask_stand"
raw_or_PC_unstand_or_stand="PC_unstand"
max_slope_d2_or_h2_regression = 1
max_residuals_d2_or_h2_regression = .1
min_rsquared_d2_or_h2_regression = .4
max_emb_dim_d2 = 14
theiler_estimate_from_space_time_traj=True
max_slope = max_slope_d2_or_h2_regression
max_residuals = max_residuals_d2_or_h2_regression
min_rsquared = min_rsquared_d2_or_h2_regression
tau = 1
emb_dim=1

file_nr=0
nr_PCs= vars.nr_PCs
nr_timepoints= vars.nr_timepoints
pattern_derivatives_output= vars.pattern_derivatives_output
method=method_rqa  # Fixed recurrence rate
thresh=thresh_rqa # Fixed recurrence rate of .05
lmin=lmin_rqa  # Minimum line length
# theiler_window=0
tiseanpath="C:/Users/KyraEvers/Documents/Tisean_3.0.0/bin"

# Get global Theiler window
theiler_window = processing.get_global_theiler_window({}, layout_der, vars.pattern_derivatives_output,
                                                      mask_unstand_or_stand,
                                                      raw_or_PC_unstand_or_stand,
                                                      suffix_of_interest='space-time-traj_theiler-estimate',
                                                      key_of_interest="theiler"
                                                      )

# Loop through filenames and for each file,
for file_nr, filename in enumerate(rawdata_bold_nifti_files):
    # for file_nr, filename in enumerate(rawdata_bold_nifti_files[0:2]):

    print("\n\nfile_nr: {}".format(file_nr))
    print("file name: {}".format(rawdata_bold_nifti_files[file_nr].filename))

    dict_ent = rawdata_bold_nifti_files[file_nr].get_entities()  # Dictionary with entities of file
    dict_ent['run'] = "0{}".format(int(dict_ent['run']))  # Make sure run has a leading zero
    dict_ent['pipeline'] = 'timeseries_analysis'

    # If we're analyzing the concatenated timeseries (run-01 + run-03), add to dictionary with entities
    if analyze_concat_ts:
        dict_ent["concat"] = "run-03_concat"

    if ((run_rqa) & (dict_ent['run'] == '01')):
        timeseries_analysis.RQA_wrapper(dict_ent, layout_der,
                                        pattern_derivatives_output=vars.pattern_derivatives_output,
                                        mask_unstand_or_stand=mask_unstand_or_stand,
                                        raw_or_PC_unstand_or_stand=raw_or_PC_unstand_or_stand,
                                        method=method_rqa,  # Fixed recurrence rate
                                        thresh=thresh_rqa,  # Fixed recurrence rate of .05
                                        theiler_window=theiler_window,
                                        lmin=lmin_rqa  # Minimum line length
                                        )

        # Plot (un)thresholded recurrence plot (still have to add timeseries to plot)
        visualisation.recurrence_plot(dict_ent, layout_der, pattern_derivatives_output=vars.pattern_derivatives_output,
                                      mask_unstand_or_stand=mask_unstand_or_stand,
                                      raw_or_PC_unstand_or_stand=raw_or_PC_unstand_or_stand,
                                      method=method_rqa,  # Fixed recurrence rate
                                      thresh=thresh_rqa,  # Fixed recurrence rate of .05
                                      theiler_window=theiler_window,
                                      lmin=lmin_rqa  # Minimum line length
                                      )

    # Compute correlation dimension and correlation entropy
    if ((compute_d2) & (dict_ent['run'] == '01')):
        timeseries_analysis.estimate_d2_h2( #tisean_d2(
            dict_ent, layout_der, vars.pattern_derivatives_output, Schaefer_ROIs_df,
                               vars.nr_PCs,
                               mask_unstand_or_stand=mask_unstand_or_stand,
                               raw_or_PC_unstand_or_stand=raw_or_PC_unstand_or_stand,
            max_slope=max_slope_d2_or_h2_regression, max_residuals=max_residuals_d2_or_h2_regression, min_rsquared=min_rsquared_d2_or_h2_regression, max_emb_dim=max_emb_dim_d2)

        # Plot estimate
        visualisation.c2_d2_h2_estimate_plot(dict_ent, layout_der, vars.pattern_derivatives_output, Schaefer_ROIs_df, vars.nr_PCs, mask_unstand_or_stand, raw_or_PC_unstand_or_stand, theiler_window=theiler_window, max_slope=max_slope_d2_or_h2_regression, max_residuals=max_residuals_d2_or_h2_regression, min_rsquared=min_rsquared_d2_or_h2_regression)

        # Plot how we derived d2 estimate: scaling regions for each embedding dimension
        visualisation.d2_or_h2_regression_plot(dict_ent, layout_der, vars.pattern_derivatives_output, Schaefer_ROIs_df, vars.nr_PCs, mask_unstand_or_stand, raw_or_PC_unstand_or_stand, d2_or_h2 ="d2", theiler_window=theiler_window, max_slope=max_slope_d2_or_h2_regression, max_residuals=max_residuals_d2_or_h2_regression, min_rsquared=min_rsquared_d2_or_h2_regression)

        # Plot how we derived h2 estimate: scaling regions for each embedding dimension
        visualisation.d2_or_h2_regression_plot(dict_ent, layout_der, vars.pattern_derivatives_output, Schaefer_ROIs_df, vars.nr_PCs, mask_unstand_or_stand, raw_or_PC_unstand_or_stand, d2_or_h2 ="h2", theiler_window=theiler_window, max_slope=max_slope_d2_or_h2_regression, max_residuals=max_residuals_d2_or_h2_regression, min_rsquared=min_rsquared_d2_or_h2_regression)

    plt.close('all')

## TISEAN RECOMMENDATIONS
# Visual inspection
# For all data which you use for the first time:
# look at the time series (amount of data, obvious artefacts, typical time scales, qualitative behaviour on short times),
# compute the distribution (histogram),
# compute the auto-correlation function ( corr).
