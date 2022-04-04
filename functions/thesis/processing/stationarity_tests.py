# For the tutorial notebook code used here, please see:
# https://github.com/statsmodels/statsmodels/blob/main/examples/notebooks/stationarity_detrending_adf_kpss.ipynb
# as well as an additional explanation on:
# https://www.analyticsvidhya.com/blog/2021/06/statistical-tests-to-check-stationarity-in-time-series-part-1/
from thesis.master.globalimports import *
# from thesis.timeseries_analysis.rqa import class_RP
from thesis import timeseries_analysis

# https://machinelearningmastery.com/time-series-data-stationary-python/#:~:text=A%20quick%20and%20dirty%20check,series%20is%20likely%20non%2Dstationary.
# from pandas import read_csv
# series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
# X = series.values
# X = np.array(timeseries[:])
# split = round(len(X) / 2)
# X1, X2 = X[0:split], X[split:]
# mean1, mean2 = X1.mean(), X2.mean()
# var1, var2 = X1.var(), X2.var()
# print('mean1=%f, mean2=%f' % (mean1, mean2))
# print('variance1=%f, variance2=%f' % (var1, var2))


# # Correcting: differencing
# Shift and np.diff give different results, think shift is better as it was used in the tutorial example
# adf_test((df["PC1"]-df["PC1"].shift(1)).dropna(), regression_type = 'c')
#
# adf_test(np.diff(timeseries), regression_type = 'c')
# adf_test(np.diff(np.diff(timeseries)), regression_type = 'c')

## Isliker & Kurths' (1993) stationarity chi-squared test -
# DOES NOT WORK, most likely due to a too short time series.
# It also did not work on the concatenated run 1 + run 3 timeseries,
# min-max scaling did not help, and standardization did not help.

# "The stationarity test proposed by Isliker & Kurths (1993)
# compares the distribution of the first half of the time series with
# the distribution of the entire time series using a chi-squared test.
# If there is no significant difference in the two distributions, we
# can consider the time series to be stationary. Here we briefly
# summarize the method for the convenience of the readers.
# We first bin the entire time series of length N using a fixed
# set of Q bins and use the resulting distribution to estimate the
# distribution that would result if the first half of the time series
# were binned using the same bins. Our estimation is then
# compared with the actual distribution of the first half using a
# chi-squared test." Mannatil et al. (2016)
# null hypothesis = stationarity timeseries
# alternative hypothesis = non-stationary timeseries

# # Run Isliker & Kurths' (1993) stationarity chi-squared test.
# nolitsa.utils.statcheck(np.array(PC_dict["PC1"]), bins = 20)
# nolitsa.utils.statcheck((PC_dict["PC1"] - np.mean(PC_dict["PC1"])) / np.std(PC_dict["PC1"], ddof=0), bins = 45)
# nolitsa.utils.statcheck(min_max_scaler(PC_dict["PC1"]), bins = 20)

# # Maybe data needs to be scaled first?
# def min_max_scaler(X, min=0, max=1):
#     X = np.array(X)
#     X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#     X_scaled = X_std * (max - min) + min
#     return X_scaled



# df_row = Schaefer_ROIs_df.iloc[0]
# pattern_derivatives_output = vars.pattern_derivatives_output
# mask_label = df_row["ROI_label"]
# dict_ent['suffix'] = "run-03_concat_" + "PCs_{}".format(mask_label)
# dict_ent['run'] = '01'
# dict_ent['extension'] = ".json"
# dict_ent['timeseries_or_figs'] = 'timeseries'
# dict_ent['mask_unstand_or_stand'] = "mask_stand"
# dict_ent['raw_or_PC_unstand_or_stand'] = "PC_unstand"

def get_global_theiler_window(dict_ent, layout_der, pattern_derivatives_output,
                mask_unstand_or_stand,
                raw_or_PC_unstand_or_stand,
                         suffix_of_interest='space-time-traj_theiler-estimate',
                         key_of_interest="theiler"
                         ):

    # Build file path to space-time data
    dict_ent['pipeline'] = 'timeseries_analysis'
    dict_ent['extension'] = '.json'
    dict_ent['suffix'] = 'PCs'
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand

    # Build general file path
    template_filepath = pathlib.Path(
        layout_der.build_path({**dict_ent, 'datatype':'func', 'subject': '*', 'session': '*', 'run': '01', 'task': 'rest', 'concat': 'run-03_concat',
                               'suffix': "{}_{}".format(dict_ent['suffix'], suffix_of_interest)
                               }, pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    # Get all file paths that belong to this subject, session and run
    filepaths_PCs = glob.glob(template_filepath)

    # Loop through files and extract quantity of interest
    values=[]
    for filepath_PC in filepaths_PCs:
        with open(filepath_PC, 'r') as file:
            input_dict = json.load(file)  # Read .json file
        values.append([input_dict[key_of_interest]])

    # Global Theiler window is maximum for all computed Theiler windows
    global_theiler_window = np.max(values)
    return int(np.ceil(global_theiler_window/10)*10) # Round up to nearest 10


def estimate_Theiler_window(dict_ent, layout_der, pattern_derivatives_output,
                mask_unstand_or_stand,
                raw_or_PC_unstand_or_stand,
                nr_timepoints
                            ):

    # Build file path to space-time data
    dict_ent['pipeline'] = 'timeseries_analysis'
    dict_ent['extension'] = '.csv'
    dict_ent['suffix'] = 'PCs'
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand

    # Prepare output file path
    filepath_spaceTimeTraj_output = pathlib.Path(
        layout_der.build_path({**dict_ent,
                               'suffix': "{}_{}".format(dict_ent['suffix'], 'space-time-traj'),
                               },
        pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    # Build output path for first_peak_idx data and theiler window estimate
    filepath_theiler_estimate = pathlib.Path(
        layout_der.build_path({**dict_ent,
                               'suffix': "{}_{}_{}".format(dict_ent['suffix'], 'space-time-traj', 'theiler-estimate'),
                               'extension': '.json'}, pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    # Read data
    data_spacetime = pd.read_csv(filepath_spaceTimeTraj_output, index_col=None)  # Read dataframe

    # Estimate Theiler window: for all percentage lines, compute first peak, and take the largest index of these
    first_peak_idx = data_spacetime.drop(columns="Delta_t").apply(find_first_peak_idx, axis=0)
    first_peak_idx = first_peak_idx.reset_index() # Change row indices to column
    first_peak_idx = first_peak_idx.rename(columns={0: "idx", "index": "perc"}) # Rename
    # Find values corresponding to first peak indices
    values = []
    for idx_name in first_peak_idx["perc"].values:
        values.append(data_spacetime[idx_name].iloc[first_peak_idx["idx"][first_peak_idx["perc"] == idx_name].values].values[0])
    first_peak_idx["value"] = values
    # Convert to float to make values JSON serializable
    first_peak_idx["idx"] = first_peak_idx["idx"].astype("float")

    # Theiler window
    theiler_window = np.max(first_peak_idx["idx"].values) + 1

    # Write output of theiler estimate
    json_dict = {}
    json_dict["first_peak_idx"] = first_peak_idx.T.reset_index().T.to_numpy().tolist()
    json_dict["theiler"] = int(theiler_window)

    # Save dictionary to .json file
    with open(filepath_theiler_estimate, 'w', encoding='utf-8') as outfile:
        json.dump(json_dict, outfile, sort_keys=True, indent=4,
                  ensure_ascii=True)  # indent makes it easier to read

    return theiler_window


def find_first_peak_idx(x, window_size=3, std_fraction=.1):
    # Estimate of first peak / decreasing steepness: where does the difference between the values (using a sliding window of 3 values) become less than .1 * sd of timeseries?
    idx = np.where(x.diff(periods=window_size).values < np.std(x) * std_fraction)[0][0] # Get first index
    return idx


# Get space-time trajectory data: for a range of Delta_t=1,...,T-1, find the radius (epsilon) with which we cover perc=[.25, .5, .75, 1] of neighbours
# Question: Don't know whether we should normalize by all possible pairs of neighbours, as you cannot cover all neighbours with a restricted range of \Delta_t, meaning that some would have NaN value. E.g. with a Delta_t = 2, it is impossible to find a radius with which we cover all N(N-1)/2 pairs.
def compute_space_time_traj(dict_ent, layout_der, pattern_derivatives_output,
                mask_unstand_or_stand,
                raw_or_PC_unstand_or_stand,
                nr_timepoints,
                distance_metric="euclidean",
                         verbose=True,
                    perc = [.25, .50, .75, 1] # Percentages to check for neighbourhood coverage
                ):

    # Build file path to data
    dict_ent['pipeline'] = 'preproc-rois'
    dict_ent['extension'] = '.json'
    dict_ent['suffix'] = 'PCs'
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand
    filepath_data = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    # Prepare output file path
    filepath_spaceTimeTraj_output = pathlib.Path(
        layout_der.build_path({**dict_ent, 'pipeline': 'timeseries_analysis',
                               'suffix': "{}_{}".format(dict_ent['suffix'], 'space-time-traj'),
                               'extension': '.csv'},
        pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    # Read data
    if dict_ent["extension"] == ".json":
        with open(filepath_data, 'r') as file:
            data_json = json.load(file)  # Read .json file

    # Convert to multi-dimensional np.array
    data = np.array(data_json['PLRNN']['data'])

    if verbose:
        print("Computing space-time trajectory data... This takes about five minutes (starting time: {})".format(datetime.datetime.now().time()))

    # Get recurrence matrix and distance matrix from data
    RP = timeseries_analysis.class_RP(data, # [nr_timepoints, nr_PCs*nr_ROIs]
                  # Leaving embedding dimension empty means no embedding is one (dim=1, tau=1),
                  method="frr", # Doesn't matter what we fill in here, as we are interested in the unthresholded matrix
                  thresh=.05, # Doesn't matter what we fill in here, as we are interested in the unthresholded matrix
                  theiler_window = 0 # Theiler window of 0 to compute the necessary Theiler window
                  )
    dist = RP.dist_mat(metric=distance_metric, theiler_window=0)

    # Set up dataframe of Delta_ts to add computations of each percentage of neighbourhood coverage to
    Delta_ts = pd.DataFrame(range(1, nr_timepoints-1), columns = ["Delta_t"])

    # Loop through percentages and apply function to range Delta_t = 1, ..., nr_timepoints-1 for each percentage; this takes about four minute
    for i in range(len(perc)):
        Delta_ts["perc_{}".format(perc[i])] = Delta_ts["Delta_t"].apply(compute_neighb_at_Delta_t, dist=dist, perc_neighs=perc[i])

    # old_time = time.time()
    # print("Pandas apply took: {}".format(old_time - time.time()))

    # Save
    Delta_ts.to_csv(filepath_spaceTimeTraj_output, index=False, header=True)

    return Delta_ts

# For a range of Delta_t, calculate the radius necessary to achieve a neighbourhood of perc = [.25, .50, .75, 1]
def compute_neighb_at_Delta_t(Delta_t, dist, perc_neighs):
    # Get diagonals from distance matrix that are t=1...Delta_t steps away to obtain neighbours within Delta_t
    neighb_Delta_t_list = [dist.diagonal(i) for i in range(1,
                                                           Delta_t + 1)]
    # FLatten list
    neighb_Delta_t = [item for sublist in neighb_Delta_t_list for item in sublist]
    # Find radius (epsilon) where a certain percentage of neighbours is covered
    eps = np.nanquantile(neighb_Delta_t, perc_neighs)

    return eps




# Wrapper for acf() function
def acf_wrapper(x, confint=None, alpha=.05, lags=None, zero=True):
    x=np.array(x)
    lags, nlags, irregular = _prepare_data_corr_plot(x, lags, zero)
    # acf has different return type based on alpha
    acf_x = statsmodels.tsa.stattools.acf(
        x,
        nlags=nlags,
        alpha=alpha,
        fft=False,
        bartlett_confint=True,
        adjusted=False,
        missing="none",
    )
    if alpha is not None:
        acf_x, confint = acf_x[:2]


    return acf_x, confint, lags


# From statsmodels (https://github.com/statsmodels/statsmodels/blob/main/statsmodels/graphics/tsaplots.py)
def _prepare_data_corr_plot(x, lags, zero):
    zero = bool(zero)
    irregular = False if zero else True
    if lags is None:
        # GH 4663 - use a sensible default value
        nobs = x.shape[0]
        lim = min(int(np.ceil(10 * np.log10(nobs))), nobs - 1)
        lags = np.arange(not zero, lim + 1)
    elif np.isscalar(lags):
        lags = np.arange(not zero, int(lags) + 1)  # +1 for zero lag
    else:
        irregular = True
        lags = np.asanyarray(lags).astype(int)
    nlags = lags.max(0)

    return lags, nlags, irregular


# Wrapper stationarity check (ADF & KPS & ZA test) for one mask
def check_stationarity_PCs_wrapper(df_row, dict_ent, layout_der, pattern_derivatives_output,  nr_PCs,mask_unstand_or_stand, raw_or_PC_unstand_or_stand, save_results_to_json = True, print_output = False, alpha_level = 0.05, regression_type="c"):
    # Read data
    mask_label = df_row["ROI_label"]
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    # Check whether we're interested in concatenated data, in which case we add upon the suffix
    dict_ent['suffix'] = "PCs_{}".format(mask_label)
    dict_ent['extension'] = ".json"
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand

    filepath_PC_data = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Read dictionary in .json file
    with open(filepath_PC_data, 'r') as file:
        PC_dict = json.load(file)  # Read .json file

    # Save PC data
    df_PCs = pd.DataFrame()#columns = ["PC{}".format(range(1, nr_PCs + 1))])
    for PC_nr in range(1, nr_PCs + 1):
        df_PCs["PC{}".format(PC_nr)] = PC_dict["PC{}".format(PC_nr)]  # Principal component timeseries

    # Get results ADF & KPSS stationary tests
    check_stationarity_results = df_PCs.apply(lambda x: check_stationarity(x.values, alpha_level=alpha_level, regression_type=regression_type), axis=0)

    if print_output:
        print("Subject {}, session {}, run {}, mask {}".format(dict_ent["subject"], dict_ent["session"], dict_ent["run"], mask_label))
        for PC_nr in range(1, nr_PCs + 1):
            print("PC{} : {}".format(PC_nr, check_stationarity_results["PC{}".format(PC_nr)]["conclusion"]))

    # Add results to .json file
    if save_results_to_json:
        # Add results to dictionary
        for PC_nr in range(1, nr_PCs + 1):
            PC_dict["PC{}_stationarity_check".format(PC_nr)] = check_stationarity_results["PC{}".format(PC_nr)]

        # Save dictionary to .json file
        with open(filepath_PC_data, 'w', encoding='utf-8') as outfile:
            json.dump(PC_dict, outfile, sort_keys=True, indent=4,
                      ensure_ascii=True)  # indent makes it easier to read

    # Return the conclusion codes of each PC
    return np.array([check_stationarity_results["PC{}".format(PC_nr)]["conclusion"] for PC_nr in range(1, nr_PCs + 1)]) #pd.DataFrame(np.transpose([check_stationarity_results["PC{}".format(PC_nr)]["conclusion"][0] for PC_nr in range(1, nr_PCs + 1)]), index=list(range(1, nr_PCs + 1)))

# Stationarity check (ADF & KPS test) for one timeseries
def check_stationarity(timeseries, alpha_level = 0.05, regression_type="c"):
    ADF = adf_test(timeseries, regression_type=regression_type)
    KPSS = kpss_test(timeseries, regression_type=regression_type)
    ZA = za_test(timeseries, regression_type=regression_type)

    # Case 1: Both tests conclude that the series is not stationary - The series is not stationary
    if ((ADF["p-value"] <= alpha_level ) & (KPSS["p-value"] > alpha_level) & (ZA["p-value"] <= alpha_level)):
        conclusion_stationary_tests = [0, "3/3 tests indicate stationarity (ADF <= alpha) & (KPSS > alpha) & (ZA <= alpha)"]
    elif ((ADF["p-value"] <= alpha_level ) & (KPSS["p-value"] > alpha_level) & (ZA["p-value"] > alpha_level)):
        conclusion_stationary_tests = [1, "2/3 tests indicate stationarity; structural break point at index {} (ADF <= alpha) & (KPSS > alpha) & (ZA > alpha)".format(int(ZA["breakpoint_idx"]))]
    # Case 2: Both tests conclude that the series is stationary - The series is stationary
    elif ((ADF["p-value"] > alpha_level ) & (KPSS["p-value"] <= alpha_level) & (ZA["p-value"] <= alpha_level)):
        conclusion_stationary_tests = [2, "2/3 tests indicate non-stationarity; no structural break point (ADF > alpha) & (KPSS <= alpha) & (ADF <= alpha)"]
    elif ((ADF["p-value"] > alpha_level ) & (KPSS["p-value"] <= alpha_level) & (ZA["p-value"] > alpha_level)):
        conclusion_stationary_tests = [3, "3/3 tests indicate non-stationarity; structural break point at index {} (ADF > alpha) & (KPSS <= alpha) & (ZA > alpha)".format(int(ZA["breakpoint_idx"]))]
    # Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
    elif ((ADF["p-value"] > alpha_level ) & (KPSS["p-value"] > alpha_level) & (ZA["p-value"] <= alpha_level)):
        conclusion_stationary_tests = [4, "2/3 tests indicate stationarity -> trend stationary; detrend timeseries to make stationary (ADF > alpha) & (KPSS > alpha) & (ADF <= alpha)"]
    elif ((ADF["p-value"] > alpha_level ) & (KPSS["p-value"] > alpha_level) & (ZA["p-value"] > alpha_level)):
        conclusion_stationary_tests = [5, "2/3 tests indicate non-stationarity -> trend stationary with structural break point at index {}; detrend timeseries to make stationary (ADF > alpha) & (KPSS > alpha) & (ZA > alpha)".format(int(ZA["breakpoint_idx"]))]
    # Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.
    elif ((ADF["p-value"] <= alpha_level ) & (KPSS["p-value"] <= alpha_level) & (ZA["p-value"] <= alpha_level)):
        conclusion_stationary_tests = [6, "2/3 tests indicate stationarity -> difference stationary; apply differencing to make series stationary (ADF <= alpha) & (KPSS <= alpha) & (ADF <= alpha)"]
    elif ((ADF["p-value"] <= alpha_level ) & (KPSS["p-value"] <= alpha_level) & (ZA["p-value"] > alpha_level)):
        conclusion_stationary_tests = [7, "2/3 tests indicate non-stationarity -> difference stationary with structural break point at index {}; apply differencing to make series stationary (ADF <= alpha) & (KPSS <= alpha) & (ZA > alpha)".format(int(ZA["breakpoint_idx"]))]

    #
    # # Case 1: Both tests conclude that the series is not stationary - The series is not stationary
    # if ((ADF["p-value"] <= alpha_level ) & (KPSS["p-value"] > alpha_level)):
    #     conclusion_stationary_tests = "stationary (ADF <= alpha) & (KPSS > alpha)"
    # # Case 2: Both tests conclude that the series is stationary - The series is stationary
    # elif ((ADF["p-value"] > alpha_level ) & (KPSS["p-value"] <= alpha_level)):
    #     conclusion_stationary_tests = "non-stationary (ADF > alpha) & (KPSS <= alpha)"
    # # Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
    # elif ((ADF["p-value"] > alpha_level ) & (KPSS["p-value"] > alpha_level)):
    #     conclusion_stationary_tests = "trend stationary; detrend timeseries to make stationary (ADF > alpha) & (KPSS > alpha)"
    # # Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.
    # elif ((ADF["p-value"] <= alpha_level ) & (KPSS["p-value"] <= alpha_level)):
    #     conclusion_stationary_tests = "difference stationary; apply differencing to make series stationary (ADF <= alpha) & (KPSS <= alpha)"

    result_stationary_tests = {"ADF_test_stat": ADF["Test Statistic"], "ADF_p-value" : ADF["p-value"], "KPSS_test_stat" : KPSS["Test Statistic"], "KPSS_p-value" : KPSS["p-value"], "ZA_test_stat": ZA["Test Statistic"], "ZA_p-value" : ZA["p-value"] ,
                               "conclusion" : conclusion_stationary_tests}
    return result_stationary_tests


## Augmented Dickey-Fuller (ADF) test
# The null hypothesis of the test is that the time series can be represented by a unit root, that it is not stationary (has some time-dependent structure). The alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary.
def adf_test(timeseries, regression_type = 'ctt'):
    # print ('Results of Dickey-Fuller Test:')
    dftest = statsmodels.tsa.stattools.adfuller(timeseries, regression = regression_type # Constant and trend order to include in regression; # “c” : constant only (default); “ct” : constant and trend; “ctt” : constant, and linear and quadratic trend; “n” : no constant, no trend.
                      ) # maxlag{None, int} = Maximum lag which is included in test, default value of 12*(nobs/100)^{1/4} is used when None.;
    # Returns :
    # adffloat = The test statistic.
    # pvaluefloat = MacKinnon’s approximate p-value based on MacKinnon (1994, 2010).
    # usedlagint = The number of lags used.
    # nobsint = The number of observations used for the ADF regression and calculation of the critical values.
    # critical valuesdict = Critical values for the test statistic at the 1 %, 5 %, and 10 % levels. Based on MacKinnon (2010).
    # icbestfloat = The maximized information criterion if autolag is not None.

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    # for key,value in dftest[4].items():
    #     dfoutput['Critical Value (%s)'%key] = value
    # print (dfoutput)
    return dfoutput

## Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.
# Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the null hypothesis that x is level or trend stationary.
def kpss_test(timeseries, regression_type="c"):
    # print("Results of KPSS Test:")
    kpsstest = statsmodels.tsa.stattools.kpss(timeseries, regression=regression_type, nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    # for key, value in kpsstest[3].items():
    #     kpss_output["Critical Value (%s)" % key] = value
    # print(kpss_output)

    # The following message may be shown:
    # InterpolationWarning: The test statistic is outside of the range of p-values available in the
    # look-up table. The actual p-value is greater than the p-value returned.
    # Which simply means that the p-value is greater than .1, which is the highest p-value available in the look-up table. The null hypothesis is not rejected.

    return kpss_output


## Zivot and Andrews Test: structural-break unit-root test. The Zivot-Andrews test tests for a unit root in a univariate process in the presence of serial correlation and a single structural break. A structural break is defined as an abrupt change involving a change in the mean or other parameters of the process.
# H0 = unit root with a single structural break
# Non-stationary if you don't reject the null hypothesis (p > .05); Stationary if you do reject the null hypothesis (p <= .05); print("Non-Stationary") if p_value > 0.05 else print("Stationary")
def za_test(timeseries, regression_type='c'):
    # za = arch.unitroot.ZivotAndrews(timeseries, trend='c')
    # print(za.summary().as_text())

    # Run test
    zatest = statsmodels.tsa.stattools.zivot_andrews(timeseries, regression=regression_type)
    # Format output
    za_output = pd.Series(
        [zatest[0], zatest[1], zatest[3],
         zatest[4] # bpidxint = The index of x corresponding to endogenously calculated break period with values in the range [0..nobs-1].
         ],
        index=["Test Statistic", "p-value", "Lags Used", "breakpoint_idx"]
    )

    return za_output
