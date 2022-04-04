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
# datalad install https://github.com/OpenNeuroDatasets/ds003059.git && cd ds003059 && git annex enableremote s3-PUBLIC && datalad get *
# mkdir C:\Users\KyraEvers\Documents\Donders\!Thesis\Data\data_preller_2018
# cd C:\Users\KyraEvers\Documents\Donders\!Thesis\Data\data_preller_2018
# datalad install https://bitbucket.org/katrinpreller/lsd-effects-on-global-brain-connectivity/downloads/subjects.LSD.all.noGSR.udvarsme.surface_gbc_mFz.dtseries.nii
# Link to github (from openneuro): https://github.com/OpenNeuroDatasets/ds003059.git;


# To upgrade pip, open Command Prompt, navigate to
# > cd C:\Users\KyraEvers\PycharmProjects\MA_thesis
# > pip install --upgrade pip

# See nilearn's fetch_neurovault([max_images, …])

# Note: if it is giving errors about not finding a package that was imported using from globalimports import *, make sure you specify the full path, i.e. from thesis.master.globalimports import *

## Import packages
import numpy as np
import vars
from thesis.master.globalimports import *
from thesis import processing
from thesis import visualisation
from thesis import general

# See nilearn's fetch_neurovault([max_images, …])
# nilearn.fetch_neurovault
# from functions.processing import save_extracted_timeseries, save_standardized_timeseries, save_PCs, plot_timeseries, \
#     create_mask_coords_dict, save_PCs_to_mat

# Parameters
extract_timeseries = False
stand_timeseries = False
extract_PCs = False
stand_PCs = False
plot_timeseries = False
plot_phase_space = False
save_md_PCs = False
check_stationarity = False
concat_runs = True
plot_stationarity = True
compute_space_time_traj=True

# First restructure rawdata directory and create derivatives directory
# general.restructure_rawdata_dir(copy_rawdata_dir=False)
# general.create_derivatives_dir() # overwrite=True Use with care! Overwrites derivatives directory

## Carhart-Harris et al. (2016)

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

# Check stationarity - clear dataframe in case it already exists
if 'df_stationarity_results' in locals():
    del df_stationarity_results
if 'df_stationarity_concat_results' in locals():
    del df_stationarity_concat_results

# Parameters
mask_unstand_or_stand="mask_stand"
raw_or_PC_unstand_or_stand="PC_unstand"
# nr_PCs=vars.nr_PCs
# nr_timepoints=vars.nr_timepoints*2
# pattern_derivatives_output=vars.pattern_derivatives_output
# file_nr=0
# save_mat = True
# save_json = True
# save_dat = True
# save_univ_eucl_norm=False

# Loop through filenames and for each file,
for file_nr, filename in enumerate(rawdata_bold_nifti_files):
    # for file_nr, filename in enumerate(rawdata_bold_nifti_files[0:2]):

    print("\n\nfile_nr: {}".format(file_nr))
    print("file name: {}".format(rawdata_bold_nifti_files[file_nr].filename))

    dict_ent = rawdata_bold_nifti_files[file_nr].get_entities()  # Dictionary with entities of file
    dict_ent['run'] = "0{}".format(int(dict_ent['run']))  # Make sure run has a leading zero
    dict_ent['pipeline'] = 'preproc-rois'
    scan = nib.load(rawdata_bold_nifti_files[file_nr].path)  # .get_fdata() # Load resting-state scan

    # # Plot anatomical data
    # visualisation.save_anat_img(dict_ent, layout_rawdata, layout_der)

    # Plot functional data and create gif
    # visualisation.save_func_gif(dict_ent, layout_rawdata, layout_der)

    # Apply mask to extract timeseries of voxels in each ROI and save in derivatives folder

    ## Extract timeseries and save to .csv in derivatives directory
    if extract_timeseries:
        print("Extracting timeseries per mask!")

        filepaths_masks = Schaefer_ROIs_df.apply(
            lambda x: processing.save_extracted_timeseries(df_row=x, masks=masks, scan=scan, dict_ent=dict_ent,
                                                           layout_der=layout_der,
                                                           pattern_derivatives_output=vars.pattern_derivatives_output),
            axis=1)

    # print("Done extracting timeseries per mask!")

    ## Inspect raw unstandardized ROI data
    # Schaefer_ROIs_df.apply(
    #     lambda x: general.inspect_data(df_row=x, masks=masks, scan=scan, dict_ent=dict_ent, layout_der=layout_der,
    #                                         pattern_derivatives_output=vars.pattern_derivatives_output), axis=1)

    ## Standardize timeseries and save to .csv in derivatives directory
    if stand_timeseries:
        print("Standardizing timeseries!")
        filepaths_stand_masks = Schaefer_ROIs_df.apply(
            lambda x: processing.save_standardized_timeseries(df_row=x, dict_ent=dict_ent,
                                                              layout_der=layout_der,
                                                              pattern_derivatives_output=vars.pattern_derivatives_output),
            axis=1)

    # print("Done standardizing timeseries!")

    ## Principal Component Analysis (PCA): Extract first three components for each ROI; run on both unstandardized and standardized timeseries
    if extract_PCs:
        print("Extracting PCs!")
        # # Unstandardized timeseries
        # filepaths_PCs = Schaefer_ROIs_df.apply(
        #     lambda x: processing.save_PCs(df_row=x, dict_ent=dict_ent,
        #                                   layout_der=layout_der,
        #                                   pattern_derivatives_output=vars.pattern_derivatives_output,
        #                                   nr_PCs=vars.nr_PCs, mask_unstand_or_stand="mask_unstand"), axis=1)

        # Standardized timeseries
        filepaths_PCs = Schaefer_ROIs_df.apply(
            lambda x: processing.save_PCs(df_row=x, dict_ent=dict_ent,
                                          layout_der=layout_der,
                                          pattern_derivatives_output=vars.pattern_derivatives_output,
                                          nr_PCs=vars.nr_PCs,
                                          mask_unstand_or_stand=mask_unstand_or_stand), axis=1)
    if stand_PCs:
        print("Standardizing PCs!")

        ## Standardize PCs
        # # Unstandardized timeseries
        # filepaths_stand_PCs = Schaefer_ROIs_df.apply(
        #     lambda x: processing.save_standardized_PCs(df_row=x, dict_ent=dict_ent,
        #                                                layout_der=layout_der,
        #                                                pattern_derivatives_output=vars.pattern_derivatives_output,
        #                                                nr_PCs=vars.nr_PCs, mask_unstand_or_stand="mask_unstand"),
        #     axis=1)

        # Standardized timeseries
        filepaths_stand_PCs = Schaefer_ROIs_df.apply(
            lambda x: processing.save_standardized_PCs(df_row=x, dict_ent=dict_ent,
                                                       layout_der=layout_der,
                                                       pattern_derivatives_output=vars.pattern_derivatives_output,
                                                       nr_PCs=vars.nr_PCs, mask_unstand_or_stand=mask_unstand_or_stand), axis=1)

    # print("Done extracting PCs!")

    # Concatenate runs, only have to do this for one run
    if ((concat_runs) & (dict_ent['run'] == "01")):
        PCs_concat = Schaefer_ROIs_df.apply(
            lambda x: processing.concat_runs(df_row=x, dict_ent=dict_ent,
                                             layout_der=layout_der,
                                             pattern_derivatives_output=vars.pattern_derivatives_output,
                                             nr_PCs=vars.nr_PCs,
                                             mask_unstand_or_stand=mask_unstand_or_stand,
                                             raw_or_PC_unstand_or_stand=raw_or_PC_unstand_or_stand
                                             ),
            axis=1)

        # Now check stationarity for concatenated runs
        if check_stationarity:
            dict_ent["concat"] = "run-03_concat"
            stationarity_results = Schaefer_ROIs_df.apply(
                lambda x: processing.check_stationarity_PCs_wrapper(df_row=x, dict_ent=dict_ent,
                                                                    layout_der=layout_der,
                                                                    pattern_derivatives_output=vars.pattern_derivatives_output,
                                                                    nr_PCs=vars.nr_PCs, mask_unstand_or_stand=mask_unstand_or_stand,
                                                                    raw_or_PC_unstand_or_stand=raw_or_PC_unstand_or_stand,
                                                                    print_output=False),
                axis=1)
            del dict_ent["concat"]

            # Flatten list and turn to dataframe
            flat_list = np.reshape([list(item) for sublist in stationarity_results for item in sublist],
                                   (-1, vars.nr_PCs * 2))
            df_flat_list = pd.DataFrame(
                flat_list)  # In format [nr_ROIs, nr_PCs * 2] because there are two entries for every PC, the code of the conclusion and the conclusion in words

            # If df_stationary_results does not exist yet, initialize
            if 'df_stationarity_concat_results' not in locals():
                df_stationarity_concat_results = df_flat_list
            # Otherwise, bind rows to df_stationary results; compute frequency table at the end
            else:
                # df_stationarity_results = df_stationarity_results.add(df_flat_list, fill_value=0)
                df_stationarity_concat_results = pd.concat([df_stationarity_concat_results, df_flat_list], axis=0)



    # Plot timeseries and principal components per ROI and create master plot of all ROIs; save figures
    if plot_timeseries:
        print("Plotting timeseries and PCs!")

        fig_timeseries = visualisation.plot_timeseries(Schaefer_ROIs_df, vars.nr_PCs, TR, dict_ent, layout_der,
                                                       vars.pattern_derivatives_output,
                                                       mask_unstand_or_stand="mask_unstand",
                                                       raw_or_PC_unstand_or_stand="raw")

        fig_timeseries = visualisation.plot_timeseries(Schaefer_ROIs_df, vars.nr_PCs, TR, dict_ent, layout_der,
                                                       vars.pattern_derivatives_output,
                                                       mask_unstand_or_stand=mask_unstand_or_stand,
                                                       raw_or_PC_unstand_or_stand="raw")

        # Plot unstandardized PCs computed on standardized timeseries
        fig_timeseries = visualisation.plot_timeseries(Schaefer_ROIs_df, vars.nr_PCs, TR, dict_ent, layout_der,
                                                       vars.pattern_derivatives_output,
                                                       mask_unstand_or_stand=mask_unstand_or_stand,
                                                       raw_or_PC_unstand_or_stand=raw_or_PC_unstand_or_stand)

        # Plot standardized PCs computed on standardized timeseries
        fig_timeseries = visualisation.plot_timeseries(Schaefer_ROIs_df, vars.nr_PCs, TR, dict_ent, layout_der,
                                                       vars.pattern_derivatives_output,
                                                       mask_unstand_or_stand=mask_unstand_or_stand,
                                                       raw_or_PC_unstand_or_stand="PC_stand")

    # Plot 3D phase space of and principal components per ROI per mask
    if plot_phase_space:
        visualisation.plot_phase_space_3D_wrapper(Schaefer_ROIs_df, dict_ent, layout_der,
                                                  vars.pattern_derivatives_output,
                                                  mask_unstand_or_stand=mask_unstand_or_stand,
                                                  raw_or_PC_unstand_or_stand=raw_or_PC_unstand_or_stand)

        visualisation.plot_phase_space_3D_wrapper(Schaefer_ROIs_df, dict_ent, layout_der,
                                                  vars.pattern_derivatives_output,
                                                  mask_unstand_or_stand=mask_unstand_or_stand,
                                                  raw_or_PC_unstand_or_stand="PC_stand")

    # print("Done plotting timeseries and PCs!")

    # Save all ROI PCs to one matlab file with the right type of structure
    if save_md_PCs:
        # First create dictionary of all mask coordinates, as this is the same in each subject+session+run
        mask_coords_dict = processing.create_mask_coords_dict(masks)
        ## Now read in all PC data and create one matlab file for this subject+session+run

        # # Unstandardized timeseries
        # processing.save_PCs_to_mat(dict_ent, layout_der, vars.pattern_derivatives_output, mask_coords_dict,
        #                            Schaefer_ROIs_df, TR, vars.nr_PCs, vars.nr_timepoints,
        #                            mask_unstand_or_stand="mask_unstand", raw_or_PC_unstand_or_stand=raw_or_PC_unstand_or_stand)
        # processing.save_PCs_to_mat(dict_ent, layout_der, vars.pattern_derivatives_output, mask_coords_dict,
        #                            Schaefer_ROIs_df, TR, vars.nr_PCs, vars.nr_timepoints,
        #                            mask_unstand_or_stand="mask_unstand", raw_or_PC_unstand_or_stand="PC_stand")

        # # Standardized timeseries
        # processing.save_PCs_to_mat(dict_ent, layout_der, vars.pattern_derivatives_output, mask_coords_dict,
        #                            Schaefer_ROIs_df, TR,
        #                            vars.nr_PCs, vars.nr_timepoints, mask_unstand_or_stand=mask_unstand_or_stand,
        #                            raw_or_PC_unstand_or_stand=raw_or_PC_unstand_or_stand)
        # processing.save_PCs_to_mat(dict_ent, layout_der, vars.pattern_derivatives_output, mask_coords_dict,
        #                            Schaefer_ROIs_df, TR,
        #                            vars.nr_PCs, vars.nr_timepoints, mask_unstand_or_stand=mask_unstand_or_stand,
        #                            raw_or_PC_unstand_or_stand="PC_stand")

        # Also save concatenated runs to multi-dimensional .mat and .json files
        if dict_ent['run'] == '01':
            dict_ent["concat"] = "run-03_concat"
            # Standardized timeseries
            processing.save_md_PCs(dict_ent, layout_der, vars.pattern_derivatives_output, mask_coords_dict,
                                   Schaefer_ROIs_df, TR,
                                   nr_PCs=vars.nr_PCs, nr_timepoints=vars.nr_timepoints*2, # A we're dealing with concatenated timeseries
                        mask_unstand_or_stand=mask_unstand_or_stand,
                                   raw_or_PC_unstand_or_stand=raw_or_PC_unstand_or_stand)
            del dict_ent["concat"]

    # if check_stationarity:
    if ((check_stationarity) & (dict_ent['run'] == '01')):
        stationarity_results = Schaefer_ROIs_df.apply(
            lambda x: processing.check_stationarity_PCs_wrapper(df_row=x, dict_ent=dict_ent,
                                                                layout_der=layout_der,
                                                                pattern_derivatives_output=vars.pattern_derivatives_output,
                                                                nr_PCs=vars.nr_PCs, mask_unstand_or_stand=mask_unstand_or_stand,
                                                                raw_or_PC_unstand_or_stand=raw_or_PC_unstand_or_stand,
                                                                print_output=False),
            axis=1)
        # Flatten list and turn to dataframe
        flat_list = np.reshape([list(item) for sublist in stationarity_results for item in sublist],
                               (-1, vars.nr_PCs * 2))
        df_flat_list = pd.DataFrame(
            flat_list)  # In format [nr_ROIs, nr_PCs * 2] because there are two entries for every PC, the code of the conclusion and the conclusion in words

        # If df_stationary_results does not exist yet, initialize
        if 'df_stationarity_results' not in locals():
            df_stationarity_results = df_flat_list
        # Otherwise, bind rows to df_stationary results; compute frequency table at the end
        else:
            # df_stationarity_results = df_stationarity_results.add(df_flat_list, fill_value=0)
            df_stationarity_results = pd.concat([df_stationarity_results, df_flat_list], axis=0)

    if ((plot_stationarity) & (dict_ent['run'] == '01')):
        visualisation.create_fig_stationary_wrapper(Schaefer_ROIs_df, vars.nr_PCs, TR, dict_ent, layout_der, vars.pattern_derivatives_output,
                                      mask_unstand_or_stand=mask_unstand_or_stand, raw_or_PC_unstand_or_stand=raw_or_PC_unstand_or_stand)


    if ((compute_space_time_traj) & (dict_ent['run'] == '01')):
        dict_ent["concat"] = "run-03_concat"
        # Compute space-time trajectory data
        processing.compute_space_time_traj(dict_ent, layout_der,
                                           pattern_derivatives_output=vars.pattern_derivatives_output,
                                           mask_unstand_or_stand=mask_unstand_or_stand,
                                           raw_or_PC_unstand_or_stand=raw_or_PC_unstand_or_stand,
                            nr_timepoints=vars.nr_timepoints*2)

        # Compute Theiler window
        theiler_window = processing.estimate_Theiler_window(dict_ent, layout_der,
                                                    pattern_derivatives_output=vars.pattern_derivatives_output,
                                           mask_unstand_or_stand=mask_unstand_or_stand,
                                           raw_or_PC_unstand_or_stand=raw_or_PC_unstand_or_stand,
                            nr_timepoints=vars.nr_timepoints*2
                                )
        print("Estimate Theiler_window: {}".format(int(theiler_window)))

        # Plot space-time trajectory
        visualisation.space_time_traj_plot(dict_ent, layout_der,
                                      pattern_derivatives_output=vars.pattern_derivatives_output,
                                      mask_unstand_or_stand=mask_unstand_or_stand,
                                      raw_or_PC_unstand_or_stand=raw_or_PC_unstand_or_stand,
                                      nr_timepoints=vars.nr_timepoints * 2
                                      )
        del dict_ent["concat"]

##########################################################

# np.sum(df_stationarity_results, axis=1)

if check_stationarity:
    ## Format df_stationarity_results
    # Rename columns
    df_stationarity_results.columns = ["PC1", "Conclusion_PC1", "PC2", "Conclusion_PC2", "PC3", "Conclusion_PC3"]
    # Add index column
    df_stationarity_results["idx"] = range(0, df_stationarity_results.shape[0])
    # Reshape to wide format
    # df_stationarity_results_wide = pd.wide_to_long(df_stationarity_results[["idx", "Code_PC1", "Code_PC2", "Code_PC3"]], stubnames='Code', i='idx',j='PC_nr', sep='_') # i=['famid', 'birth'],
    df_stationarity_results_wide = pd.melt(df_stationarity_results[["idx", "PC1", "PC2", "PC3"]], id_vars=['idx'],
                                           var_name='PC_nr', value_name='Code')

    # Get frequency table
    df_freq = pd.crosstab(index=df_stationarity_results_wide['Code'], columns=df_stationarity_results_wide['PC_nr'])
    print(df_freq)
    print(np.sum(df_freq, axis=1) / np.sum(np.sum(df_freq)))

    # Check what codes mean
    df_stationarity_results[["PC1", "Conclusion_PC1"]].drop_duplicates("PC1").to_dict('dict')  # .iloc[columns=[0,1]]
    df_conclusion_codes = df_stationarity_results[["PC1", "Conclusion_PC1"]].drop_duplicates(
        "PC1")  # .to_dict('list')#.iloc[columns=[0,1]]

    dict_conclusion_codes = dict(zip(df_conclusion_codes.PC1, df_conclusion_codes.Conclusion_PC1))
    print(dict_conclusion_codes)  # {'a': 1, 'b': 2, 'c': 3}
    [dict_conclusion_codes[key] for key in df_freq.index]


    ##########################################################
    df_stationarity_concat_results.columns = ["PC1", "Conclusion_PC1", "PC2", "Conclusion_PC2", "PC3", "Conclusion_PC3"]
    # Add index column
    df_stationarity_concat_results["idx"] = range(0, df_stationarity_concat_results.shape[0])
    # Reshape to wide format
    df_stationarity_concat_results_wide = pd.melt(df_stationarity_concat_results[["idx", "PC1", "PC2", "PC3"]], id_vars=['idx'],
                                           var_name='PC_nr', value_name='Code')

    # Get frequency table
    df_concat_freq = pd.crosstab(index=df_stationarity_concat_results_wide['Code'], columns=df_stationarity_concat_results_wide['PC_nr'])
    print(df_concat_freq)
    print(np.sum(df_concat_freq, axis=1) / np.sum(np.sum(df_concat_freq)))

    print("As show above, the number of non-stationary timeseries when the timeseries are concatenated is only two.")
##########################################################

## TISEAN RECOMMENDATIONS
# Visual inspection
# For all data which you use for the first time:
# look at the time series (amount of data, obvious artefacts, typical time scales, qualitative behaviour on short times),
# compute the distribution (histogram),
# compute the auto-correlation function ( corr).
