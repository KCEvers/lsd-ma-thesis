
# Import packages
from thesis.master.globalimports import *


# Average modified permutation entropy (EntropyHub)


def compute_entropies(dict_ent, layout_der, pattern_derivatives_output, Schaefer_ROIs_df, nr_PCs, mask_unstand_or_stand, raw_or_PC_unstand_or_stand, theiler_window=0, tau = 1, template_length_SampEn=3, r_fraction_SampEn=0.2, template_length_PermEn=4):

    # Build input file path
    dict_ent['pipeline'] = 'preproc-rois'
    dict_ent['extension'] = '.json'
    dict_ent['suffix'] = 'PCs'
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand

    # Output file path
    filepath_SampEn = pathlib.Path(
        layout_der.build_path({**dict_ent,
                               'pipeline': 'timeseries_analysis',
                               'suffix': "{}_{}".format(dict_ent['suffix'], 'sampEn')
    }, pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    filepath_PermEn = pathlib.Path(
        layout_der.build_path({**dict_ent,
                               'pipeline': 'timeseries_analysis',
                               'suffix': "{}_{}".format(dict_ent['suffix'], 'permEn')
    }, pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    # Read .json file:
    filepath_json = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()
    with open(filepath_json, 'r') as file:
        data_json = json.load(file)  # Read .json file

    # Convert to dataframe
    df = pd.DataFrame(data_json['PLRNN']['data']) # [nr_timepoints, nr_PCs*nr_ROIs]

    # Compute sample entropy
    SampEns = compute_SampEn(df).T.reset_index().T.to_numpy().tolist()

    # Compute permutation entropy
    PermEns = compute_PermEn(df, template_length=template_length_PermEn, tau=tau).T.reset_index().T.to_numpy().tolist()

    # Write to .json file
    with open(filepath_SampEn, 'w', encoding='utf-8') as outfile:
        json.dump(dict(SampEn=SampEns), outfile, sort_keys=True, indent=4,
                  ensure_ascii=True)  # indent makes it easier to read

    # Write to .json file
    with open(filepath_PermEn, 'w', encoding='utf-8') as outfile:
        json.dump(dict(SampEn=PermEns), outfile, sort_keys=True, indent=4,
                  ensure_ascii=True)  # indent makes it easier to read

    return None

def compute_SampEn(data, # [nr_timepoints, nr_PCs*nr_ROIs]
                   tolerances = [.2,.3,.4,.5],
                   template_lengths=[2,3]):

    # Apply sample entropy function to all template lengths
    list_SampEn_r_dfs = list(
        map(functools.partial(SampEn_r_wrapper, tolerances=tolerances, df=df),
            template_lengths))  # Apply function to all tolerances
    # Bind rows to create one dataframe
    SampEn_r_dfs = pd.concat([entry for entry in list_SampEn_r_dfs], axis=0)
    return SampEn_r_dfs

# Compute sample entropy for fixed m and range of tolerances
def SampEn_r_wrapper(template_length, tolerances, df):
    ## Apply function to range of tolerances
    list_SampEn_dfs = list(
        map(functools.partial(EH_SampEn_wrapper, df=df, template_length=template_length),
            tolerances))  # Apply function to all tolerances
    # Bind rows to create one dataframe
    SampEn_dfs = pd.concat([entry for entry in list_SampEn_dfs], axis=0)
    return SampEn_dfs

# Apply function to dataframe
def EH_SampEn_wrapper(r_fraction, df, template_length=3):
    # SampEns = pd.DataFrame(df.apply(lambda x: EH_SampEn(x, r_fraction=.2, template_length=3), axis=0), columns=["SampEn"])
    SampEns = df.apply(lambda x: EH_SampEn(x, r_fraction=.2, template_length=3), axis=0).transpose().rename(columns={0:"SampEn_norm", 1:"A", 2:"B"})
    SampEns["m"] = template_length # Add template length column
    SampEns["r_fraction"] = r_fraction # Add fraction of std used as tolerance
    SampEns["r"] = r_fraction * df.apply(np.std) # Add actual tolerance used per timeseries
    SampEns = SampEns.reset_index().rename(columns={'index': 'ROI_nr'})  # Add column with ROI indices
    return SampEns

def EH_SampEn(df_col, r_fraction=.2, template_length=3,
              norm=True # Return normalized sample entropy
              ):
    # Question: Normalize time series before?
    r = r_fraction*np.std(df_col.values) # Tolerance
    SampleEntropy, A, B = EH.SampEn(df_col.values, m=template_length, tau=1, r=r) # SampleEntropy = -np.log(A/B)

    # Normalize sample entropy
    if norm:
        N=len(df_col.values) # Length timeseries
        SampleEntropy = SampleEntropy * (1 / (-1 * np.log(1 / N))) # Normalize by maximum entropy

    return SampleEntropy[-1], A[-1], B[-1] # Only return sample entropy for last m (contains sample entropy for entire vector 0,...,m
    # return SampleEntropy[-1] # Only return sample entropy for last m (contains sample entropy for entire vector 0,...,m


def compute_PermEn(df, template_length=4,tau=1):
    # EntropyHub and ordpy give the same result! Only ordpy has the option of computing statistical complexity as well, so using ordpy

        # X = df[0].values
        # X=np.random.normal(size=400)
        # Perm, Pnorm, cPE = EH.PermEn(X, m=template_length, tau=tau, Norm=True)
        # Perm, Pnorm, cPE = EH.PermEn(X, m=3)#, m=template_length, tau=tau)#, Norm=True)
        # Pnorm[-1]
        # Returns the permutation entropy estimates (Perm), the normalised permutation entropy (Pnorm) and the conditional permutation entropy (cPE) for m = [1,2] estimated from the data sequence (Sig)

    # Loop through columns and compute permutation entropy for each timeseries separately
    permEns = []
    statComps = []
    for nr_col in range(df.shape[1]):
        X = df[nr_col].values.tolist()
        permutationEntropy, statisticalComplexity = ordpy.complexity_entropy(
            X,  # df.transpose().to_numpy().tolist(), # [nr_ROIs*nr_PCs, nr_timepoints]
            # df.drop(columns="t").transpose().values.tolist(),
            # Data should be nested list with M number of entries each of length N, M being the number of variables and N being the number of timesteps
            dx=template_length,  # Embedding dimension (horizontal axis) (default: 3)
            # dy=1,  # Embedding dimension (vertical axis); it must be 1 for time series (default: 1)
            # probs=False, # Data entered is not a vector of probabilities
            # taux=1, # Embedding delay (horizontal axis) (default: 1).
            tauy=tau,  # Embedding delay (vertical axis) (default: 1).
        )  # Returns (normalized) permutation entropy and statistical complexity; exact same as the function
        permEns.append(permutationEntropy)
        statComps.append(statisticalComplexity)

    PermEns_df = pd.DataFrame(np.transpose([permEns, statComps]), columns=["PermEn_norm", "Statisical_Complexity"])
    PermEns_df["m"] = template_length  # Add template length column
    PermEns_df["tau"] = tau  # Add actual tolerance used per timeseries
    PermEns_df = PermEns_df.reset_index().rename(columns={'index': 'ROI_nr'})  # Add column with ROI indices

    # np.mean(permEns_df["PermEn_norm"])
    #
    # # Average permutation entropy (ordpy; multivariate in that you compute it over multiple timeseries):
    # permutationEntropy, statisticalComplexity = ordpy.complexity_entropy(
    #     X.tolist(),#df.transpose().to_numpy().tolist(), # [nr_ROIs*nr_PCs, nr_timepoints]
    #     # df.drop(columns="t").transpose().values.tolist(),
    #     # Data should be nested list with M number of entries each of length N, M being the number of variables and N being the number of timesteps
    #     dx=template_length,  # Embedding dimension (horizontal axis) (default: 3)
    #     # dy=1,  # Embedding dimension (vertical axis); it must be 1 for time series (default: 1)
    #     # probs=False, # Data entered is not a vector of probabilities
    #     # taux=1, # Embedding delay (horizontal axis) (default: 1).
    #     tauy=tau, # Embedding delay (vertical axis) (default: 1).
    #
    # )  # Returns permutation entropy and statistical complexity; exact same as the function
    # permutationEntropy # Returns normalized value by default
    #
    # statisticalComplexity
    #
    #
    # # So the mean perm en over all timeseries is not the same as the perm en computed over the entire dataframe...
    # # Probs because of the log?
    #
    # symbols, probabilities = ordpy.ordinal_distribution(df.transpose().to_numpy().tolist(), dx=template_length, dy=1, taux=1, tauy=1, return_missing=True, tie_precision=None)
    # symbols
    # probabilities

        # # Compute permutation entropy for entire dataframe (computes composite permutation entropy)
        # # Average permutation entropy (ordpy; multivariate in that you compute it over multiple timeseries):
        # permutationEntropy, statisticalComplexity = ordpy.complexity_entropy(
        #     df.transpose().to_numpy().tolist(), # [nr_ROIs*nr_PCs, nr_timepoints]
        #     # df.drop(columns="t").transpose().values.tolist(),
        #     # Data should be nested list with M number of entries each of length N, M being the number of variables and N being the number of timesteps
        #     dx=template_length,  # Embedding dimension (horizontal axis) (default: 3)
        #     # dy=1,  # Embedding dimension (vertical axis); it must be 1 for time series (default: 1)
        #     # probs=False, # Data entered is not a vector of probabilities
        #     # taux=1, # Embedding delay (horizontal axis) (default: 1).
        #     tauy=tau, # Embedding delay (vertical axis) (default: 1).
        #
        # )  # Returns permutation entropy and statistical complexity; exact same as the function
        # permutationEntropy # Returns normalized value by default




        # Normalize
        # (np.log(np.math.factorial(k + 1)) / np.log(Logx))
    #     To allow every possible order pattern of dimension m to occur in a time series of length N, the condition m! ≤ N − (m − 1)l must hold. Moreover, to avoid undersampling, N ≥ m! + (m − 1)l is required. Therefore, we need to choose N ≥ (m + 1)!. For N = 130, an obviously unsatisfying complexity estimation is obtained when m ≥ 5. To satisfy this condition, we therefore chose a low dimension, i.e., m = 4
    #     m = 4
    #     N = 434
    #     tau = 1
    # #     m! ≤ N − (m − 1)l
    #     math.factorial(m) <= N - (m - 1) * tau
    # #     N ≥ m! + (m − 1)l
    #     N >= math.factorial(m) + (m-1)*tau
    # #     N ≥ (m + 1)!
    #     N >= math.factorial(m+1)




        # m must be minimally 6 because N=434 and with m=5, m! + (m − 1)l=124, with m=6, m! + (m − 1)l=725

    return PermEns_df






































import rpy2

import rpy2.robjects as robjects


import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr

utils = rpackages.importr('utils')
MSMVSampEn = importr("MSMVSampEn")

utils.chooseCRANmirror(ind=1) # select the first mirror in the list
packnames = ('MSMVSampEn')#, 'hexbin')
from rpy2.robjects.vectors import StrVector
utils.install_packages(StrVector(packnames))

# areshenk/MSMVSampEn

# If you are a user of bioconductor:

utils.chooseBioCmirror(ind=1) # select the first mirror in the list

print(robjects.r('1+2'))

sqr = robjects.r('function(x) x^2')
print(sqr)
print(sqr(2))

x = robjects.r.rnorm(100)
robjects.r('hist(%s, xlab="x", main="hist(x)")' %x.r_repr())


