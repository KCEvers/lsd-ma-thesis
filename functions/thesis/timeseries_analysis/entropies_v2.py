
# Import packages
from thesis.master.globalimports import *


# Average modified permutation entropy (EntropyHub)


def compute_entropies(dict_ent, layout_der, pattern_derivatives_output, Schaefer_ROIs_df, nr_PCs, mask_unstand_or_stand, raw_or_PC_unstand_or_stand, theiler_window=0, tau = 10, emb_dim=1):

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

    # nr_ROIs = Schaefer_ROIs_df.shape[0]

    # Compute sample entropy
    SampEns = compute_SampEn(df).T.reset_index().T.to_numpy().tolist()

    # Compute permutation entropy
    PermEns = compute_PermEn(df).T.reset_index().T.to_numpy().tolist()
    compute_PermEn(data, tau = tau, emb_dim=emb_dim)

    # Write to .json file
    with open(filepath_SampEn, 'w', encoding='utf-8') as outfile:
        json.dump(dict(SampEn=SampEns), outfile, sort_keys=True, indent=4,
                  ensure_ascii=True)  # indent makes it easier to read

    # Write to .json file
    with open(filepath_PermEn, 'w', encoding='utf-8') as outfile:
        json.dump(dict(SampEn=PermEns), outfile, sort_keys=True, indent=4,
                  ensure_ascii=True)  # indent makes it easier to read

    return

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
    SampEns = pd.DataFrame(df.apply(lambda x: EH_SampEn(x, r_fraction=.2, template_length=3), axis=0), columns=["SampEn"])
    SampEns["m"] = template_length # Add template length column
    SampEns["r_fraction"] = r_fraction # Add fraction of std used as tolerance
    SampEns["r"] = r_fraction * df.apply(np.std) # Add actual tolerance used per timeseries
    SampEns = SampEns.reset_index().rename(columns={'index': 'ROI_nr'})  # Add column with ROI indices
    return SampEns

def EH_SampEn(df_col, r_fraction=.2, template_length=3):
    # Question: Normalize time series before?
    r = r_fraction*np.std(df_col.values) # Tolerance
    SampleEntropy, A, B = EH.SampEn(data_col.values, m=template_length, tau=1, r=r) # SampleEntropy = -np.log(A/B)
    return SampleEntropy[-1] # Only return sample entropy for last m (contains sample entropy for entire vector 0,...,m

        # Only use one m
        # check whether sample entropy is the same when time series is standardized

        # Only use one m
        # check whether sample entropy is the same when time series is standardized
        ts = [1, 4, 5, 1, 7, 3, 1, 2, 5, 8, 9, 7, 3, 7, 9, 5, 4, 3]
        ts = np.random.normal(size=400)
        ts
        ts_stand = (ts - np.mean(ts)) / np.std(ts)
        std_ts = np.std(ts)

        SampleEntropy, A, B = EH.SampEn(ts, m=3, tau=1, r=.2 * std_ts)  # SampleEntropy = -np.log(A/B)
        SampleEntropy
        SampleEntropy, A, B = EH.SampEn(ts_stand, m=3, tau=1, r=.2 * std_ts)  # SampleEntropy = -np.log(A/B)
        SampleEntropy
        # not the same....

        sample_entropy = ent.sample_entropy(ts, 4, 0.2 * std_ts)
        #

        ts = [1, 4, 5, 1, 7, 3, 1, 2, 5, 8, 9, 7, 3, 7, 9, 5, 4, 3]
        std_ts = np.std(ts)
        sample_entropy = ent.sample_entropy(ts, 4, 0.2 * std_ts)
        #


        # # SampleEntropy, A, B = EH.SampEn(data[0], m=0, tau=1) # SampleEntropy = -np.log(A/B)
        #
        #
        # from pyentrp import entropy as ent
        # #
        # ts = [1, 4, 5, 1, 7, 3, 1, 2, 5, 8, 9, 7, 3, 7, 9, 5, 4, 3]
        # std_ts = np.std(ts)
        # sample_entropy = ent.sample_entropy(ts, 4, 0.2 * std_ts)
        # #
        # import antropy as ant
        # np.random.seed(1234567)
        # x = np.random.normal(size=3000)
        # # # Permutation entropy
        # # print(ant.perm_entropy(x, normalize=True))
        # # # Approximate entropy
        # # print(ant.app_entropy(x))
        # # # Sample entropy
        # print(ant.sample_entropy(x, order=6, metric='euclidean'))
        #
        # # Returns the sample entropy estimates (Samp) and the number of matched state vectors (m: B, m+1: A) for m = [0, 1, 2] estimated from the data sequence (Sig) using the default parameters: embedding dimension = 2, time delay = 1, radius threshold = 0.2*SD(Sig), logarithm = natural
        #
        # SampleEntropy



def compute_PermEn():
    if package == "EntropyHub":
        permutationEntropy = EH.PermEn(X, m=emb_dim, tpx="")
        permutationEntropy
        # Returns the permutation entropy estimates (Perm), the normalised permutation entropy (Pnorm) and the conditional permutation entropy (cPE) for m = [1,2] estimated from the data sequence (Sig)

    if package == "ordpy":
        # Average permutation entropy (ordpy; multivariate in that you compute it over multiple timeseries):
        permutationEntropy, statisticalComplexity = ordpy.complexity_entropy(
            df.drop(columns="t").transpose().values.tolist(),
            # Data should be nested list with M number of entries each of length N, M being the number of variables and N being the number of timesteps
            # dx=3,  # Embedding dimension (horizontal axis) (default: 3)
            # dy=1  # Embedding dimension (vertical axis); it must be 1 for time series (default: 1)
        )  # Returns permutation entropy and statistical complexity; exact same as the function
        # ordpy.permutation_entropy(df.drop(columns="t").transpose().values.tolist(), # Data should be nested list with M number of entries each of length N, M being the number of variables and N being the numer of timestep,
        #                           dx=3,
        #                           # dy=1, taux=1, tauy=1, base=2, normalized=True, probs=False, tie_precision=None
        #                           )

    return permutationEntropy






































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


