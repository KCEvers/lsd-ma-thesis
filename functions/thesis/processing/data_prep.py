
"""
Functions for data preparation: extract timeseries, standardize, compute principal components.
"""



# Import packages
from thesis.master.globalimports import *


# Concatenate run 1 and run 3
def concat_runs(df_row, dict_ent, layout_der, pattern_derivatives_output, nr_PCs, mask_unstand_or_stand, raw_or_PC_unstand_or_stand):
    # Run 1
    if "concat" in dict_ent.keys():
        del dict_ent["concat"]
    dict_ent['run'] = '01'
    mask_label = df_row["ROI_label"]
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['suffix'] = "PCs_{}".format(mask_label)
    dict_ent['extension'] = ".json"
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand

    filepath_PC_data = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True))
    # PC data run 1
    # Read dictionary in .json file
    with open(filepath_PC_data, 'r') as file:
        PC_dict_run1 = json.load(file)  # Read .json file

    # Run 3; same parameters, only different run number
    dict_ent['run'] = '03'
    filepath_PC_data = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True))
    # PC data run 3
    # Read dictionary in .json file
    with open(filepath_PC_data, 'r') as file:
        PC_dict_run3 = json.load(file)  # Read .json file

    # Concatenate
    PCs_dict_concat = {}
    # Concatenate PC{} from run 1 and run 3 along temporal axis
    for PC_nr in range(1,nr_PCs+1):
        PCs_dict_concat["PC{}".format(PC_nr)] = list(PC_dict_run1["PC{}".format(PC_nr)] + PC_dict_run3["PC{}".format(PC_nr)])

    # Add ROI label
    PCs_dict_concat['ROI_label'] = PC_dict_run1['ROI_label']

    # Add explained variance
    PCs_dict_concat['explained_variance_ratio_'] = {"run-01": PC_dict_run1['explained_variance_ratio_'], "run-03": PC_dict_run3['explained_variance_ratio_']}

    # Write to .json file
    # Filepath to write .json file to
    dict_ent["run"] = "01"
    dict_ent["concat"] = "run-03_concat" #+ dict_ent['suffix']
    filepath_concat_PCs = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Save dictionary to .json file
    with open(filepath_concat_PCs, 'w', encoding='utf-8') as outfile:
        json.dump(PCs_dict_concat, outfile, sort_keys=True, indent=4, ensure_ascii=True)  # indent makes it easier to read

    return filepath_concat_PCs


def create_mask_coords_dict(masks):
    mask_coords_dict = {}
    for mask in masks.keys():
        mask_coords = np.transpose(np.transpose(np.nonzero(masks[mask].get_fdata()))) # Need to apply double transpose to get data in format where rows are x,y,z coordinates and columns are voxels
        mask_coords_dict[mask] = mask_coords

    return mask_coords_dict


# def save_PCs_to_mat(dict_ent, layout_der, pattern_derivatives_output, mask_coords_dict, Schaefer_ROIs_df, TR, nr_PCs, nr_timepoints, mask_unstand_or_stand, raw_or_PC_unstand_or_stand):
#
#     # Prepare for dictionary that will be saved to .mat file
#     nr_ROIs = Schaefer_ROIs_df.shape[0]
#     all_PCs = np.zeros((nr_timepoints, nr_ROIs*nr_PCs))# [PCs x time]
#
#     ROI_labels_with_PC_nr = np.zeros((nr_ROIs*nr_PCs,), dtype=object) # Needs to be an object to be saved as a cell structure
#     ROI_coords = np.zeros((nr_ROIs*nr_PCs,), dtype=object) # Needs to be an object to be saved as a cell structure
#     # template_ROI[0] = {mask_label}
#     # template_ROI[1] = {}
#
#     # Read all PCs from all ROIs
#     # Build path by filling in dict_ent (making sure to include the appropriate suffix and extension) in pattern_derivatives_output to find PCs files in derivatives directory
#     dict_ent['suffix'] = 'PCs_*'
#     dict_ent['extension'] = '.json'
#     dict_ent['timeseries_or_figs'] = 'timeseries'
#     dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
#     dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand
#
#     # Build general file path
#     template_filepath_PCs = pathlib.Path(
#         layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
#                               absolute_paths=True)).as_posix()
#
#     # Get all file paths that belong to this subject, session and run
#     filepaths_PCs = glob.glob(template_filepath_PCs)
#
#     # For each of these filepaths, read their data and add to dictionary that will be saved to .mat file
#     i = 0
#     for filepath_PCs in filepaths_PCs:
#
#         # Read dictionary in .json file
#         with open(filepath_PCs, 'r') as file:
#             masked_data_dict = json.load(file)  # Read .json file
#         ROI_label = masked_data_dict["ROI_label"]
#         # print("ROI_label: {}".format(ROI_label))
#
#         # Save PC data
#         for PC_nr in range(1, nr_PCs+1):
#             all_PCs[:,i] = masked_data_dict["PC{}".format(PC_nr)] # Principal component timeseries
#             # ROI_labels_with_PC_nr[i] = {ROI_label + "_PC1"}
#             ROI_labels_with_PC_nr[i] = ROI_label + "PC{}".format(PC_nr) # ROI label (same for both PCs except for suffix _PC*)
#             ROI_coords[i] = mask_coords_dict[ROI_label].astype(int) # ROI coordinates (same for both PCs)
#             i += 1 # Add to counter
#
#     # Create final PLRNN dictionary for .mat file
#     PLRNN = {"PLRNN": {"data": all_PCs,
#                        "rp": [], # No nuisance parameters
#                        "preprocess": {"RT": TR}, # Repetition time
#                        "ROI": {"labs": ROI_labels_with_PC_nr,
#                                "coords": ROI_coords,
#                                }
#
#                        }
#              }
#
#     # Save file - Build path by filling in dict_ent
#     dict_ent['suffix'] = 'PCs'
#     dict_ent['extension'] = '.mat'
#     dict_ent['timeseries_or_figs'] = 'timeseries'
#
#     # Build file path to .mat file
#     filepath_mat = pathlib.Path(
#         layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
#                               absolute_paths=True)).as_posix()
#
#     # Save dictionary
#     scipy.io.savemat(filepath_mat, PLRNN)
#
#     return filepath_mat


# Save multi-dimensional PC file
def save_md_PCs(dict_ent, layout_der, pattern_derivatives_output, mask_coords_dict, Schaefer_ROIs_df, TR, nr_PCs, nr_timepoints, mask_unstand_or_stand, raw_or_PC_unstand_or_stand, save_mat = True, save_json = True, save_dat = True, save_univ_eucl_norm=True):

    # Prepare for dictionary that will be saved to .mat file
    nr_ROIs = Schaefer_ROIs_df.shape[0]
    all_PCs = np.zeros((nr_timepoints, nr_ROIs*nr_PCs))# [PCs x time]

    ROI_labels_with_PC_nr = np.zeros((nr_ROIs*nr_PCs,), dtype=object) # Needs to be an object to be saved as a cell structure
    ROI_coords = np.zeros((nr_ROIs*nr_PCs,), dtype=object) # Needs to be an object to be saved as a cell structure
    # template_ROI[0] = {mask_label}
    # template_ROI[1] = {}

    # Read all PCs from all ROIs
    # Build path by filling in dict_ent (making sure to include the appropriate suffix and extension) in pattern_derivatives_output to find PCs files in derivatives directory
    dict_ent['pipeline'] = 'preproc-rois'
    dict_ent['suffix'] = 'PCs_7Networks_*'
    dict_ent['extension'] = '.json'
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand

    # Build general file path
    template_filepath_PCs = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    # Get all file paths that belong to this subject, session and run
    filepaths_PCs = glob.glob(template_filepath_PCs)

    # For each of these filepaths, read their data and add to dictionary that will be saved to .mat file
    i = 0
    for filepath_PCs in filepaths_PCs:

        # Read dictionary in .json file
        with open(filepath_PCs, 'r') as file:
            masked_data_dict = json.load(file)  # Read .json file
        ROI_label = masked_data_dict["ROI_label"]
        # print("ROI_label: {}".format(ROI_label))

        # Save PC data
        for PC_nr in range(1, nr_PCs+1):
            all_PCs[:,i] = masked_data_dict["PC{}".format(PC_nr)] # Principal component timeseries
            # ROI_labels_with_PC_nr[i] = {ROI_label + "_PC1"}
            ROI_labels_with_PC_nr[i] = ROI_label + "PC{}".format(PC_nr) # ROI label (same for both PCs except for suffix _PC*)
            ROI_coords[i] = mask_coords_dict[ROI_label].astype(int) # ROI coordinates (same for both PCs)
            i += 1 # Add to counter

    # Create final PLRNN dictionary for .mat file
    PLRNN = {"PLRNN": {"data": all_PCs,
                       "rp": [], # No nuisance parameters
                       "preprocess": {"RT": TR}, # Repetition time
                       "ROI": {"labs": ROI_labels_with_PC_nr,
                               "coords": ROI_coords,
                               }

                       }
             }

    # Format in way that is acceptable to .json
    PLRNN_json_format = copy.deepcopy(PLRNN)
    PLRNN_json_format["PLRNN"]["data"] = PLRNN_json_format["PLRNN"]["data"].tolist() # [nr_timepoints, nr_PC]s
    PLRNN_json_format["PLRNN"]["ROI"]["coords"] = [entry.tolist() for entry in PLRNN_json_format["PLRNN"]["ROI"]["coords"]]#.tolist()
    # for entry in PLRNN_json_format["PLRNN"]["ROI"]["coords"]:
    #     print(entry.tolist())
    PLRNN_json_format["PLRNN"]["ROI"]["labs"] = PLRNN_json_format["PLRNN"]["ROI"]["labs"].tolist()

    # Save file - Build path by filling in dict_ent
    dict_ent['suffix'] = 'PCs'
    dict_ent['extension'] = '.mat'
    dict_ent['timeseries_or_figs'] = 'timeseries'

    # Build file path to .mat file
    filepath_mat = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Save dictionary
    if save_mat:
        scipy.io.savemat(filepath_mat.as_posix(), PLRNN)

    if save_json:
        with open(filepath_mat.with_suffix(".json"), 'w', encoding='utf-8') as outfile:
            json.dump(PLRNN_json_format, outfile, sort_keys=True, indent=4,
                      ensure_ascii=True)  # indent makes it easier to read

    if save_dat:

        ## Prepare input for tisean; convert to .dat file
        # tisean requires the .dat format, where they used two spaces as a separator. Recreate this format:
        float_to_string = ["  ".join(map(str, (entry))) for entry in all_PCs]
        dat_formatted_string = "  " + "\n  ".join(
            float_to_string) + "\n"  # Add two empty spaces to beginning of string and join string together by \n, finish string with new line which seems to be crucial for tisean.d2 being able to read the file!

        # Write to .dat file
        with open(filepath_mat.with_suffix(".dat"), 'w') as your_dat_file:
            your_dat_file.write(dat_formatted_string)

    # Save multivariate data as univariate sequence using the Euclidean norm
    if ((save_univ_eucl_norm) & ("concat" in dict_ent.keys())):

        eucl_norm = np.linalg.norm(all_PCs, axis = 1) # [nr_timepoints*2, nr_PCs*nr_ROIs] -> [nr_timepoints*2]
        # Format suitable for .dat
        float_to_string = "\n".join(map(str, eucl_norm)) + "\n"

        # Write to .dat file
        with open(filepath_mat.with_name(filepath_mat.stem + "_euclnorm" + ".dat"), 'w') as your_dat_file:
            your_dat_file.write(float_to_string)

    return filepath_mat

# def read_PLRNN_mat():
#     dict_data = scipy.io.loadmat(filepath_mat) # Is the data dictionary for .mat files
#     sel_vars = ["Ephizi", "Ephizij", "Ezi", "Eziphizj", "LL", "V", "None"]
#     # [dict_data[key] for key in sel_vars]
#
#     timings_columns = timings_mat.dtype  # A structure is read by scipy as a list of arrays with column names or keys as dtype
#     timings_dict = {n: timings_mat[n][0, 0] for n in timings_columns.names}  # Create dictionary of timings
#     # Get keys of timing vectors (i.e. those with length equal to the number of trials)
#     timings_keys = [n for n, v in timings_dict.items() if v.size == nr_trials]
#
#     return

def save_standardized_PCs(df_row, dict_ent, layout_der, pattern_derivatives_output, nr_PCs, mask_unstand_or_stand, same_as_matlab_std=False):
    ## Read PC data
    # Build path by filling in dict_ent (making sure to include the appropriate suffix and extension) in pattern_derivatives_output to save timeseries file in derivatives directory
    mask_label = df_row["ROI_label"]
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['suffix'] = "PCs_{}".format(mask_label)
    dict_ent['extension'] = ".json"
    dict_ent['raw_or_PC_unstand_or_stand'] = "PC_unstand"

    filepath_unstand_PC_data = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Read dictionary in .json file
    with open(filepath_unstand_PC_data, 'r') as file:
        PC_unstand_dict = json.load(file)  # Read .json file

    PC_stand_dict = copy.deepcopy(PC_unstand_dict)

    # Standardize
    # Matlab's std() uses n-1 degrees of freedom, whereas np.std() uses n degrees of freedom (biased estimator). If you want the same result as in matlab, use np.std(, ddof=1)
    for PC_nr in range(1,nr_PCs+1):
        if same_as_matlab_std:
            PC_stand_dict["PC{}".format(PC_nr)] = list((PC_unstand_dict["PC{}".format(PC_nr)] - np.mean(PC_unstand_dict["PC{}".format(PC_nr)])) / np.std(PC_unstand_dict["PC{}".format(PC_nr)], ddof=1))

        elif not same_as_matlab_std:  # use default np.std(, ddof=0)
            PC_stand_dict["PC{}".format(PC_nr)] = list((PC_unstand_dict["PC{}".format(PC_nr)] - np.mean(PC_unstand_dict["PC{}".format(PC_nr)])) / np.std(PC_unstand_dict["PC{}".format(PC_nr)], ddof=0))

    # Filepath to write .json file to
    dict_ent['raw_or_PC_unstand_or_stand'] = "PC_stand"
    filepath_stand_PC_data = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Save dictionary to .json file
    with open(filepath_stand_PC_data, 'w', encoding='utf-8') as outfile:
        json.dump(PC_stand_dict, outfile, sort_keys=True, indent=4, ensure_ascii=True)  # indent makes it easier to read

    return filepath_stand_PC_data


def save_PCs(df_row, dict_ent, layout_der, pattern_derivatives_output, nr_PCs, mask_unstand_or_stand, same_as_matlab_std=False):
    # df_row = Schaefer_ROIs_df.iloc[0,:]

    ## Read masked data
    # Build path by filling in dict_ent (making sure to include the appropriate suffix and extension) in pattern_derivatives_output to save timeseries file in derivatives directory
    # df_row = Schaefer_ROIs_df.iloc[2]
    mask_label = df_row["ROI_label"]
    dict_ent['suffix'] = "mask_{}".format(mask_label)
    dict_ent['extension'] = '.csv'
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['raw_or_PC_unstand_or_stand'] = 'raw'

    filepath_masked_data = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True))
    # Read data
    masked_data = pd.read_csv(filepath_masked_data, sep='\t')  # Read csv file

    # # Check for NaN or infinite values
    # np.sum(np.sum(np.isinf(masked_data)))
    # np.sum(np.sum(np.isnan(masked_data)))
    # np.sum(np.isnan(masked_data), axis=1)
    # masked_data.iloc[0].values
    # idx_nan = np.where(np.isnan(masked_data.iloc[0].values))
    #
    # masked_data.iloc[0].values[idx_nan]

    # Run PCA
    pca = sklearn.decomposition.PCA(n_components=nr_PCs)
    PCs = pca.fit_transform(X=masked_data)  # Fit to data ([observations, features] = 217, nr_voxels) and extract principal components

    ## Save PCs
    # Build dictionary with first two principal components and explained variance
    PCs_dict = {'ROI_label': mask_label,
                'explained_variance_ratio_': list(pca.explained_variance_ratio_),
                # 'PC1': list(PCs[:, 0]),
                # 'PC2': list(PCs[:, 1])
    }  # The values need to be converted to lists to be json serializable

    # Add PC timeseries to dictionary
    for PC_nr in range(1,nr_PCs+1):
        PCs_dict["PC{}".format(PC_nr)] = list(PCs[:, PC_nr-1])

    # Build path by filling in dict_ent (making sure to include the appropriate suffix and extension) in pattern_derivatives_output to save timeseries file in derivatives directory
    dict_ent['suffix'] = "PCs_{}".format(mask_label)
    dict_ent['extension'] = ".json"
    dict_ent['raw_or_PC_unstand_or_stand'] = "PC_unstand"

    filepath_PCs = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Save dictionary to .json file
    with open(filepath_PCs, 'w', encoding='utf-8') as outfile:
        json.dump(PCs_dict, outfile, sort_keys=True, indent=4, ensure_ascii=True)  # indent makes it easier to read

    return filepath_PCs


def save_standardized_timeseries(df_row, dict_ent, layout_der, pattern_derivatives_output, same_as_matlab_std=False):
    """ Standardize data by removing the mean and scaling to unit variance. The standard score of a sample x is calculated as: z = (x - u) / s, with mean u and standard deviation s.

    :return:
    """

    # mask_label = Schaefer_ROIs_df["ROI_label"][0]

    # Build path by filling in dict_ent (making sure to include the appropriate suffix and extension) in pattern_derivatives_output to save timeseries file in derivatives directory
    mask_label = df_row["ROI_label"]
    dict_ent['suffix'] = "mask_{}".format(mask_label)
    dict_ent['extension'] = ".csv"
    dict_ent['timeseries_or_figs'] = "timeseries"
    dict_ent['mask_unstand_or_stand'] = "mask_unstand"
    dict_ent['raw_or_PC_unstand_or_stand'] = "raw"

    # Build file path to unstandardized data
    filepath_masked_data = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Read data
    masked_data_df = pd.read_csv(filepath_masked_data, sep='\t')  # Read csv file

    ## Standardize data - first fit standardization and then apply it to data
    # Sklearn's StandardScaler() standardizes each column independently by for each column, subtracting the mean of the column from the column's values and subsequently dividing the column's values by the standard deviation of the column. As a result, the new mean is zero and the new standard deviation is 1.
    # The mean, variance, and standard deviation of all columns can be extracted from the data using
    # >> scaler.mean_
    # >> scaler.var_
    # >> scaler.scale_

    # Note that sklearn gives the same standardized result as using same_as_matlab_std = False below: namely, with np.std(, ddof=0), the default setting

    # The operation can also be done in one go using fit_transform()
    # masked_data_stand = sklearn.preprocessing.StandardScaler().fit_transform(masked_data_df)
    # masked_data_stand = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(masked_data_df),
    #                                  columns=masked_data_df.columns)

    # Matlab's std() uses n-1 degrees of freedom, whereas np.std() uses n degrees of freedom (biased estimator). If you want the same result as in matlab, use np.std(, ddof=1)
    if same_as_matlab_std:
        masked_data_stand = ((masked_data_df - masked_data_df.mean(axis=0)) / masked_data_df.std(axis=0, ddof=1))

    elif not same_as_matlab_std:  # use default np.std(, ddof=0)
        masked_data_stand = ((masked_data_df - masked_data_df.mean(axis=0)) / masked_data_df.std(axis=0, ddof=0))

    # Set any NaN values to 0 - these are the consequence of dividing 0 by something
    masked_data_stand[np.isnan(masked_data_stand)] = 0

    # masked_data_stand.mean(axis=0)
    # masked_data_stand.iloc[:, 0].mean()

    # Build path by filling in dict_ent (making sure to include the appropriate suffix and extension) in pattern_derivatives_output to save standardized timeseries file in derivatives directory
    dict_ent['suffix'] = "mask_{}".format(mask_label)
    dict_ent['extension'] = ".csv"
    dict_ent['timeseries_or_figs'] = "timeseries"
    dict_ent['mask_unstand_or_stand'] = "mask_stand"
    dict_ent['raw_or_PC_unstand_or_stand'] = "raw"

    filepath_stand_masked_data = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Save to csv file
    masked_data_stand.to_csv(filepath_stand_masked_data, sep='\t', encoding='ascii', index=False, header=True)
    return filepath_stand_masked_data


def save_extracted_timeseries(df_row, masks, scan, dict_ent, layout_der, pattern_derivatives_output):
    # Extract timeseries of each voxel in mask
    mask_label = df_row["ROI_label"]
    print("Extracting timeseries for mask {} (start: {})".format(mask_label, datetime.datetime.now()))

    masked_data = extract_timeseries(masks[mask_label], scan)  # Returns [time x voxels]

    # Get voxel coordinates of ROI voxels and turn them into strings ("x,y,z") to use them as column names
    voxel_coords = np.transpose(np.nonzero(masks[mask_label].get_fdata()))  # np.nonzero() removes the zero vectors
    voxel_coords_names = list(map(lambda x: ",".join(str(i) for i in x), list(voxel_coords)))
    masked_data_df = pd.DataFrame(masked_data,
                                  columns=voxel_coords_names)  # Turn masked data into dataframe [time x voxels]

    # Build path by filling in dict_ent (making sure to include the appropriate suffix and extension) in pattern_derivatives_output to save timeseries file in derivatives directory
    dict_ent['suffix'] = "mask_{}".format(mask_label)
    dict_ent['extension'] = ".csv"
    dict_ent['timeseries_or_figs'] = "timeseries"
    dict_ent['mask_unstand_or_stand'] = "mask_unstand"
    dict_ent['raw_or_PC_unstand_or_stand'] = "raw"

    filepath_masked_data = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Save to csv file
    masked_data_df.to_csv(filepath_masked_data, sep='\t', encoding='ascii', index=False, header=True)
    return filepath_masked_data


def extract_timeseries(mask, scan):
    return nilearn.masking.apply_mask(scan,
                                      mask)  # Apply masking through time; Timeseries of only the voxels in the mask, with different voxels in columns and timeseries across rows, i.e. [time x voxels]













