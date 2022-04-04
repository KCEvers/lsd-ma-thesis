# Import packages
from python.master.globalimports import *



def noise_reduction_wrapper():

    return

def noise_reduction():

    # Have to apply noise reduction to each time series separately

    # Run ghkss()
    output = subprocess.run([r".\ghkss",
                             # "chaotic_lor_01_stand_raw.dat",
                             input_path_dat,
                             # "-v0.06",
                             # "-r2",
                             # "-d{}".format(tau),
                             # -q  # dimension of the manifold to project to	2
                             # - k  # minimal number of neighbours	30
                             # - r  # minimal size of the neighbourhood	(interval of data)/1000
                             # "-m8",
                             "-i3",  # number of iterations
                             # "-t{}".format(int(theiler_window)),
                             # "-M1,{}".format(emb_dim),
                             "-c2",
                             # "-c1,2,3", # Multivariate
                             # "-m", # Multivariate
                             # "-V128",  # Verbosity level
                             "-o"
                             # "-o{}".format(output_path)
                             ], cwd=tiseanpath, shell=True,
                            universal_newlines=True, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # Necessary to get output.stdout
    output.stderr


    return


def tisean_lyap(dict_ent, layout_der, pattern_derivatives_output, Schaefer_ROIs_df, nr_PCs, mask_unstand_or_stand, raw_or_PC_unstand_or_stand, tiseanpath="C:/Users/KyraEvers/Documents/Tisean_3.0.0/bin", theiler_window=0, tau = 1, emb_dim=1
              ):

    # Build input file path
    dict_ent['pipeline'] = 'preproc-rois'
    dict_ent['extension'] = '.dat'
    dict_ent['suffix'] = 'PCs'
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand
    input_path_dat = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    # Prepare output path - remove extension as there will be four different outputs
    output_path_full = pathlib.Path(
        layout_der.build_path({**dict_ent,
                               'pipeline':'timeseries_analysis',
                               'suffix': "{}_{}".format(dict_ent['suffix'], 'd2output')},
                              pattern_derivatives_output, validate=False,
                              absolute_paths=True))
    output_path = pathlib.Path(output_path_full.parent, output_path_full.stem).as_posix()

    nr_ROIs = Schaefer_ROIs_df.shape[0]

    # ## Prepare input for tisean; convert to .dat file
    # # Read dictionary in .json file
    # if pathlib.Path(input_path).suffixes[0] == ".json":
    #     with open(input_path, 'r') as file:
    #         input_dict = json.load(file)  # Read .json file
    # input_list = [input_dict[var] for var in vars_of_interest]
    # input_df = pd.DataFrame(np.transpose(input_list), # Transpose to get a dataframe of [t, vars]
    #              columns = vars_of_interest) # Convert to dataframe
    #
    # # tisean requires the .dat format, where they used two spaces as a separator. Recreate this format:
    # float_to_string = ["  ".join(map(str,(entry))) for entry in np.transpose(input_list)]
    # dat_formatted_string = "  " + "\n  ".join(float_to_string) + "\n" # Add two empty spaces to beginning of string and join string together by \n, finish string with new line which seems to be crucial for tisean.d2 being able to read the file!
    #
    # # Write to .dat file
    # input_path_dat = pathlib.Path(input_path).with_suffix(".dat")
    # with open(input_path_dat, 'w') as your_dat_file:
    #     your_dat_file.write(dat_formatted_string)

    # Specify which columns (according to format -c1,2,3) you want to read; use all
    which_columns = "-c" + ",".join([str(i) for i in list(range(1, (nr_ROIs*nr_PCs)+1))])

    # Run lyap_k()
    # left off here - did not edit command from d2() example
    output = subprocess.run([".\lyap_k",
                             # "chaotic_lor_01_stand_raw.dat",
                             input_path_dat,
                             # "-d{}".format(tau),
                             "-t{}".format(int(theiler_window)),
                             # "-M1,{}".format(emb_dim),
                             which_columns,
                             # "-c1,2,3", # Multivariate
                             # "-m", # Multivariate
                             # "-V128",  # Verbosity level
                             # "-o"
                             "-o{}".format(output_path)
                             ], cwd=tiseanpath, shell=True,
                            universal_newlines=True, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # Necessary to get output.stdout
    output.stderr





    return


def tisean_d2(dict_ent, layout_der, pattern_derivatives_output, Schaefer_ROIs_df, nr_PCs, mask_unstand_or_stand, raw_or_PC_unstand_or_stand, tiseanpath="C:/Users/KyraEvers/Documents/Tisean_3.0.0/bin", theiler_window=0, tau = 1, emb_dim=1
              ):

    # Build input file path
    dict_ent['pipeline'] = 'preproc-rois'
    dict_ent['extension'] = '.dat'
    dict_ent['suffix'] = 'PCs'
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand
    input_path_dat = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    # Prepare output path - remove extension as there will be four different outputs
    output_path_full = pathlib.Path(
        layout_der.build_path({**dict_ent,
                               'pipeline':'timeseries_analysis',
                               'suffix': "{}_{}_theiler_{}".format(dict_ent['suffix'], 'd2output', theiler_window)},
                              pattern_derivatives_output, validate=False,
                              absolute_paths=True))
    output_path = pathlib.Path(output_path_full.parent, output_path_full.stem).as_posix()

    nr_ROIs = Schaefer_ROIs_df.shape[0]

    # ## Prepare input for tisean; convert to .dat file
    # # Read dictionary in .json file
    # if pathlib.Path(input_path).suffixes[0] == ".json":
    #     with open(input_path, 'r') as file:
    #         input_dict = json.load(file)  # Read .json file
    # input_list = [input_dict[var] for var in vars_of_interest]
    # input_df = pd.DataFrame(np.transpose(input_list), # Transpose to get a dataframe of [t, vars]
    #              columns = vars_of_interest) # Convert to dataframe
    #
    # # tisean requires the .dat format, where they used two spaces as a separator. Recreate this format:
    # float_to_string = ["  ".join(map(str,(entry))) for entry in np.transpose(input_list)]
    # dat_formatted_string = "  " + "\n  ".join(float_to_string) + "\n" # Add two empty spaces to beginning of string and join string together by \n, finish string with new line which seems to be crucial for tisean.d2 being able to read the file!
    #
    # # Write to .dat file
    # input_path_dat = pathlib.Path(input_path).with_suffix(".dat")
    # with open(input_path_dat, 'w') as your_dat_file:
    #     your_dat_file.write(dat_formatted_string)

    # Specify which columns (according to format -c1,2,3) you want to read; use all
    which_columns = "-c" + ",".join([str(i) for i in list(range(1, (nr_ROIs*nr_PCs)+1))])

    # Run d2()
    output = subprocess.run([".\d2",
                             # "chaotic_lor_01_stand_raw.dat",
                             input_path_dat,
                             # "-d{}".format(tau),
                             "-t{}".format(int(theiler_window)),
                             # "-M1,{}".format(emb_dim),
                             which_columns,
                             # "-c1,2,3", # Multivariate
                             # "-m", # Multivariate
                             # "-V128",  # Verbosity level
                             # "-o"
                             "-o{}".format(output_path)
                             ], cwd=tiseanpath, shell=True,
                            universal_newlines=True, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # Necessary to get output.stdout
    output.stderr

    # Convert output d2(): Clean each entry and convert to dataframe; bind all dataframes
    list_d2_dfs = list(map(functools.partial(convert_d2_output, output_path=output_path),
                   [".c2", # Correlation sums
                    ".d2", # Correlation dimension
                    ".h2" # Correlation entropy
                    ]) # Apply function to all file extensions
               )

    # Merge dataframes according to embedding dimension and epsilon
    d2_df = list_d2_dfs[0] # Starting dataframe
    for idx in range(1, (len(list_d2_dfs))):
        d2_df = pd.merge(d2_df, list_d2_dfs[idx], how='outer', left_on=['epsilon', 'emb_nr'], right_on=['epsilon', 'emb_nr'])
    d2_df

    # Convert all data types to float
    d2_df = d2_df.astype("float")

    # Write final dataframe with all d2 outputs to .csv file
    d2_df.to_csv(pathlib.Path(output_path).with_suffix(".csv"), index=False, header=True, sep=",")

    return d2_df

# Plot output tisean.d2() with scaling region
def estimate_d2_h2(dict_ent, layout_der, pattern_derivatives_output, Schaefer_ROIs_df, nr_PCs, mask_unstand_or_stand, raw_or_PC_unstand_or_stand, theiler_window=0, tau = 1, emb_dim=1,
                   # scaling_region=[10**(-.8), 10**0],
                   max_emb_dim = 14,
                   max_slope=1, max_residuals=.1, min_rsquared=.4
                   ):
    # (1/tau) -> "Do not forget to divide the h2-estimate by the time lag" (https://www.pks.mpg.de/tisean//Tisean_3.0.1/docs/tutorial/ex4.html)
    # Check whether this is necessary still!
    #  Question: same scaling region for correlation entropy?


    # left off here
    #     Correlation entropy plateau? -> different estimate!!! /check whether estimate works in same way and plot
    # Next up: nonlinear noise reduction

    # Build input file path
    dict_ent['pipeline'] = 'timeseries_analysis'
    dict_ent['extension'] = '.csv'
    dict_ent['suffix'] = 'PCs'
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand

    # Define tisean d2 output file path to read data from
    filepath_d2_output = pathlib.Path(
        layout_der.build_path({**dict_ent,
                               'suffix': "{}_{}_theiler_{}".format(dict_ent['suffix'], 'd2output', theiler_window)},
                              pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Define file path for output d2 estimate
    filepath_d2_h2_estimate = pathlib.Path(
        layout_der.build_path({**dict_ent,
                               'extension': '.json',
                               'suffix': "{}_{}_theiler_{}".format(dict_ent['suffix'], 'd2-h2-estimate', theiler_window)},
                              pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Run tisean.d2() to get d2 and h2 estimates for all embedding dimensions
    tisean_d2(dict_ent, layout_der, pattern_derivatives_output, Schaefer_ROIs_df,
                                  nr_PCs,
                                  mask_unstand_or_stand,
                                  raw_or_PC_unstand_or_stand
)
    # Read dataframe
    d2_df = pd.read_csv(filepath_d2_output, index_col=None) # Read dataframe

    # Select up to a certain embedding dimension and drop columns not being used
    # d2_df_select = d2_df[d2_df["emb_nr"] < max_emb_dim].drop(columns=["c2", "h2"]).sort_values(by="epsilon")
    d2_df_select = d2_df[d2_df["emb_nr"] < max_emb_dim].drop(columns=["h2"]).sort_values(by="epsilon")
    h2_df_select = d2_df[d2_df["emb_nr"] < max_emb_dim].drop(columns=["d2"]).sort_values(by="epsilon")

    d2_eps_scaling_region_dfs, d2_model_dfs, d2_model_scaling_dfs, d2_plateau_dfs = find_plateau_through_regression_wrapper(d2_df_select.drop(columns="c2"), d2_or_h2 = "d2", max_slope=max_slope, max_residuals=max_residuals, min_rsquared=min_rsquared)
    h2_eps_scaling_region_dfs, h2_model_dfs, h2_model_scaling_dfs, h2_plateau_dfs = find_plateau_through_regression_wrapper(h2_df_select.drop(columns="c2"), d2_or_h2 = "h2", max_slope=max_slope, max_residuals=max_residuals, min_rsquared=min_rsquared)

    # Compute d2 estimate and scaling region (the mean of all beginnings of plateaus and the mean of all endings of plateaus)
    # d2_estimate = np.mean(model_scaling_dfs["d2"])
    d2_estimate = np.mean(d2_model_scaling_dfs["Intercept"])
    d2_eps_scaling_region_min = np.mean(d2_eps_scaling_region_dfs["min_eps_scaling_region"])
    d2_eps_scaling_region_max = np.mean(d2_eps_scaling_region_dfs["max_eps_scaling_region"])

    h2_estimate = np.mean(h2_model_scaling_dfs["Intercept"])
    h2_eps_scaling_region_min = np.mean(h2_eps_scaling_region_dfs["min_eps_scaling_region"])
    h2_eps_scaling_region_max = np.mean(h2_eps_scaling_region_dfs["max_eps_scaling_region"])

    ## Save to .json file
    # Add column headers as row to make sure they are preserved in the list; thne convert to numpy, then convert to list
    json_dict={}
    json_dict["global_d2_estimate"] = d2_estimate
    json_dict["global_d2_eps_scaling_region"] = [d2_eps_scaling_region_min, d2_eps_scaling_region_max]
    json_dict["d2_df_select"] = d2_df_select.T.reset_index().T.to_numpy().tolist()
    json_dict["d2_eps_scaling_region_dfs"] = d2_eps_scaling_region_dfs.T.reset_index().T.to_numpy().tolist()
    json_dict["d2_model_dfs"] = d2_model_dfs.T.reset_index().T.to_numpy().tolist()
    json_dict["d2_model_scaling_dfs"] = d2_model_scaling_dfs.T.reset_index().T.to_numpy().tolist()
    json_dict["d2_plateau_dfs"] = d2_plateau_dfs.T.reset_index().T.to_numpy().tolist()

    json_dict["global_h2_estimate"] = h2_estimate
    json_dict["global_h2_eps_scaling_region"] = [h2_eps_scaling_region_min, h2_eps_scaling_region_max]
    json_dict["h2_df_select"] = h2_df_select.T.reset_index().T.to_numpy().tolist()
    json_dict["h2_eps_scaling_region_dfs"] = h2_eps_scaling_region_dfs.T.reset_index().T.to_numpy().tolist()
    json_dict["h2_model_dfs"] = h2_model_dfs.T.reset_index().T.to_numpy().tolist()
    json_dict["h2_model_scaling_dfs"] = h2_model_scaling_dfs.T.reset_index().T.to_numpy().tolist()
    json_dict["h2_plateau_dfs"] = h2_plateau_dfs.T.reset_index().T.to_numpy().tolist()

    # Save dictionary to .json file
    with open(filepath_d2_h2_estimate, 'w', encoding='utf-8') as outfile:
        json.dump(json_dict, outfile, sort_keys=True, indent=4,
                  ensure_ascii=True)  # indent makes it easier to read

    plt.close('all')
    # Check estimate: for a timeseries of length N, dimension estimates larger than 2 * \log10{N} cannot be justified -> Mannatil et al. (2016) citing Eckmann & Ruelle (1992)
    # N = 434
    # assert d2_estimate < 2 * np.log10(N)  # 2 * np.log10(434) = 5.275

    # For noise, the d2_estimate is equal to the embedding dimension
    # assert np.allclose()

    return None

def find_plateau_through_regression_wrapper(df_select,
                                    d2_or_h2="d2",  # Identify scaling region for d2 or h2?
                                    max_slope=1, max_residuals=.1, min_rsquared=.4, window_size=5):

    ## Find plateaus using rolling regression
    list_plateau_dfs = list(
        map(functools.partial(find_plateau_through_regression, df_select=df_select, d2_or_h2=d2_or_h2),
            # Remove period from file extension to use file extension use as column name
            np.unique(df_select["emb_nr"]))  # Apply function to all elements of chunked list
    )

    # Find edges of scaling region
    eps_scaling_region_dfs = pd.DataFrame(np.transpose(pd.concat([entry[0] for entry in list_plateau_dfs], axis=1)))
    eps_scaling_region_dfs.columns = ["emb_nr", "min_eps_scaling_region", "max_eps_scaling_region"]
    model_dfs = pd.concat([entry[1] for entry in list_plateau_dfs], axis=0)
    model_scaling_dfs = pd.concat([entry[2] for entry in list_plateau_dfs], axis=0)
    plateau_dfs = pd.concat([entry[3] for entry in list_plateau_dfs], axis=0)

    return eps_scaling_region_dfs, model_dfs, model_scaling_dfs, plateau_dfs

def find_plateau_through_regression(emb_nr, df_select,
                                    d2_or_h2 = "d2", # Identify scaling region for d2 or h2?
                                    max_slope = 1, max_residuals = .1, min_rsquared = .4, window_size = 5):

    # Get dataframe with values only for particular embedding dimension
    d2_df_emb = df_select[df_select.emb_nr == emb_nr].dropna(subset=[d2_or_h2, "epsilon"]).sort_values(by="epsilon")
    d2_df_emb = d2_df_emb.reset_index(drop=True)  # Reset index without adding new column
    # Add np.log(epsilon)
    d2_df_emb['log_epsilon'] = np.log(d2_df_emb.epsilon)  # Add constant (needs to be done manually)

    # Conduct rolling regression
    model = statsmodels.regression.rolling.RollingOLS.from_formula('{} ~ log_epsilon'.format(d2_or_h2), data=d2_df_emb, window=window_size)
    rres = model.fit() # Fit model

    # The slope log_epsilon should approximately be zero
    model_df = pd.concat([rres.params, rres.mse_resid, rres.rsquared], axis=1)
    model_df = model_df.rename(columns={0: "residuals", 1: "r-squared", "log_epsilon": "slope"})
    model_df["emb_nr"] = emb_nr # Add column for embedding dimension
    model_df["epsilon"] = d2_df_emb["epsilon"][model_df.index] # Add corresponding epsilon values
    model_df[d2_or_h2] = d2_df_emb[d2_or_h2][model_df.index] # Add corresponding d2 values

    # Find rows in model dataframe for which the slope is .., the residuals are .., and r-squared is
    model_df_scaling = model_df[((np.abs(model_df["slope"]) < max_slope) & (model_df["residuals"] < max_residuals) & (model_df["r-squared"] > min_rsquared))]

    # D2 estimate
    # d2_estimate = np.mean(model_df_scaling["Intercept"])

    # Scaling region
    eps_scaling_region = pd.Series([emb_nr, np.min(model_df_scaling["epsilon"]), np.max(model_df_scaling["epsilon"])])

    return eps_scaling_region, model_df, model_df_scaling, d2_df_emb.iloc[model_df_scaling.index]


    # Old function, only for d2:
    # def find_plateau_through_regression(emb_nr, d2_df_select, max_slope = 1, max_residuals = .1, min_rsquared = .4, window_size = 5):
    #
    #     # Get dataframe with values only for particular embedding dimension
    #     d2_df_emb = d2_df_select[d2_df.emb_nr == emb_nr].dropna(subset=["d2", "epsilon"]).sort_values(by="epsilon")
    #     d2_df_emb = d2_df_emb.reset_index(drop=True)  # Reset index without adding new column
    #     # Add np.log(epsilon)
    #     d2_df_emb['log_epsilon'] = np.log(d2_df_emb.epsilon)  # Add constant (needs to be done manually)
    #
    #     # Conduct rolling regression
    #     model = statsmodels.regression.rolling.RollingOLS.from_formula('d2 ~ log_epsilon', data=d2_df_emb, window=window_size)
    #     rres = model.fit() # Fit model
    #
    #     # The slope log_epsilon should approximately be zero
    #     model_df = pd.concat([rres.params, rres.mse_resid, rres.rsquared], axis=1)
    #     model_df = model_df.rename(columns={0: "residuals", 1: "r-squared", "log_epsilon": "slope"})
    #     model_df["emb_nr"] = emb_nr # Add column for embedding dimension
    #     model_df["epsilon"] = d2_df_emb["epsilon"][model_df.index] # Add corresponding epsilon values
    #     model_df["d2"] = d2_df_emb["d2"][model_df.index] # Add corresponding d2 values
    #
    #     # Find rows in model dataframe for which the slope is .., the residuals are .., and r-squared is
    #     model_df_scaling = model_df[((np.abs(model_df["slope"]) < max_slope) & (model_df["residuals"] < max_residuals) & (model_df["r-squared"] > min_rsquared))]
    #
    #     # D2 estimate
    #     # d2_estimate = np.mean(model_df_scaling["Intercept"])
    #
    #     # Scaling region
    #     eps_scaling_region = pd.Series([emb_nr, np.min(model_df_scaling["epsilon"]), np.max(model_df_scaling["epsilon"])])
    #
    #     return eps_scaling_region, model_df, model_df_scaling, d2_df_emb.iloc[model_df_scaling.index]



def find_plateau(x, window_size=3, std_fraction=.1):
    # Estimate of first peak / decreasing steepness: where does the difference between the values (using a sliding window of 3 values) become less than .1 * sd of timeseries?
    idx = np.where(x.diff(periods=window_size).values < np.std(x) * std_fraction)[0][0] # Get first index
    return idx


def convert_d2_output(file_ext = [".c2", ".d2", ".h2"][0], output_path=""):
    # Get output file with right extension
    output_path_file_ext = pathlib.Path(output_path).with_suffix(file_ext)

    # Read output
    with open(output_path_file_ext) as file:
        d2_output = file.readlines()

    # The quantity of interest (correlation sum/dimension/entropy) is given for each embedding dimension; find indices of #emb in list
    indices_dim = [i for i, s in enumerate(d2_output) if '#dim=' in s] + [len(d2_output)] # Find indices of occurrences of "#dim=" and append length of list

    # Split list by these indices
    chunked_list = [d2_output[indices_dim[i]:indices_dim[i+1]] for i in range(len(indices_dim)-1)]

    # Clean each entry and convert to dataframe; bind all dataframes
    list_emb_dfs = list(map(functools.partial(d2_emb_to_df, chunked_list=chunked_list, file_ext=file_ext.replace(".", "")), # Remove period from file extension to use file extension use as column name
                   range(len(chunked_list))) # Apply function to all elements of chunked list
               )

    d2_df = pd.concat([entry for entry in list_emb_dfs])

    return d2_df


def d2_emb_to_df(emb_nr, chunked_list, file_ext):
    split_list = [entry.split() for entry in chunked_list[emb_nr][1:]] # Skip first entry which contains the dimension but use it as another column

    # Convert lists to dataframe; filter out empty lists
    df_emb = pd.DataFrame(filter(None, split_list), columns = ["epsilon", file_ext])
    df_emb["emb_nr"] = emb_nr #chunked_list[emb_nr][0] # Add column with embedding dimension

    return df_emb


def correct_format(t):
    return int(np.array([t]).flatten())

