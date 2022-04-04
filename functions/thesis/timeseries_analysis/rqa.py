# Recurrence Quantification Analysis (RQA)
# Please note that this is a slightly adapted version of RECLAC package https://github.com/ToBraun/RECLAC/blob/main/RECLAC/recurrence_plot.py; added features are:
# - entropy of the diagonal/vertical line length distribution
# - number of diagonal/vertical lines
# - unthresholded recurrence matrix
#
# # import RECLAC
# #
# # import RECLAC.recurrence_plot as rec
# # import RECLAC.boxcount as bc
# fs = 18
#
# a_data = np.load('C:/Users/KyraEvers/Downloads/exmpl_time_series.npy')
#
# fig, ax = plt.subplots(nrows=6, figsize=(14, 10))
# for i in range(6):
#     ax[i].plot(a_data[i,], color='navy')
#     ax[i].grid()
#     ax[i].set_ylabel('$y(t)$', fontsize=fs)
#     if i<5:
#         ax[i].set_xticklabels([])
#     else:
#         ax[i].set_xlabel('$t$', fontsize=fs)
#
# # Define a recurrence plot object from the white noise time series
# RP = rec.RP(a_data[1,], method='frr', thresh=.1)
#
# # extract the recurrence matrix via the attribute rm():
# a_rm_wn = RP.rm()
# RP.metric
# RP.rm()
#
# RP.R
#
#
#
# # plot it
# a_ticks = np.arange(0, 1200, 200)
# fig = plt.figure(figsize=(8,8))
# plt.imshow(a_rm_wn, origin='lower', cmap='binary')
# plt.xlabel('$t$', fontsize=fs)
# plt.ylabel('$t$', fontsize=fs)
# plt.xticks(a_ticks, fontsize=fs-2);
# plt.yticks(a_ticks, fontsize=fs-2);
#
# # create a list of RP objects:
# l_RP = [rec.RP(a_data[i,], method='frr', thresh=.1) for i in range(6)]
# # the Roessler time series needs to be embedded:
# l_RP[2] = rec.RP(a_data[2,], dim=3, tau=14, method='frr', thresh=.1)
#
# # Extract the recurrence matrices:
# l_rm = [l_RP[i].rm() for i in range(6)]
# fig, ax_array = plt.subplots(2, 3, squeeze=False, figsize=(18, 12))
# k = 0
# for i, ax_row in enumerate(ax_array):
#     for j, axes in enumerate(ax_row):
#         axes.imshow(l_rm[k], origin='lower', cmap='binary')
#         axes.set_xticklabels(np.arange(0, 1200, 200), fontsize=fs-2);
#         axes.set_yticklabels(np.arange(0, 1200, 200), fontsize=fs-2);
#         axes.set_xlabel("$t$", fontsize=fs)
#         if k == 0 or k == 3: axes.set_ylabel("$t$", fontsize=fs)
#         k += 1
# plt.show()



# Import packages
from thesis.master.globalimports import *

def RQA_wrapper(dict_ent, layout_der, pattern_derivatives_output,
                mask_unstand_or_stand,
                raw_or_PC_unstand_or_stand,
                method='frr', # Fixed recurrence rate
                thresh=.05, # Fixed recurrence rate of .05
                metric = 'euclidean',
                theiler_window=0,
                lmin=2 # Minimum line length
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
    filepath_output = pathlib.Path(
        layout_der.build_path({**dict_ent, 'pipeline': 'timeseries_analysis',
                               'suffix': "{}_{}_{}{}_theiler_{}".format(dict_ent['suffix'], 'RQAoutput', method, thresh, theiler_window)},
        pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    # Read data
    if dict_ent["extension"] == ".json":
        with open(filepath_data, 'r') as file:
            data_json = json.load(file)  # Read .json file

    # Convert to multi-dimensional np.array
    data = np.array(data_json['PLRNN']['data'])

    # Create multi-dimensional RP
    RP = class_RP(data, # (time, dimension)
                # Leaving embedding dimension empty means no embedding is one (dim=1, tau=1),
                method=method,
                thresh=thresh,
                  metric=metric,
                  theiler_window=theiler_window)

    # Get unthresholded matrix
    RP_unthresh = RP.dist_mat(RP.metric, RP.theiler_window)

    # Extract the thresholded recurrence matrix via the attribute rm():
    RP_thresh = RP.rm()

    # Get RQA measures
    RQA_meas = RP.RQA(lmin=lmin, measures = 'all', border=None)

    # Save to .json file
    # Convert to float format so it the dictionary is JSON serializable
    for key in RQA_meas.keys():
        RQA_meas[key] = np.float64(RQA_meas[key])
    RQA_output = {**RQA_meas,
                  "method": method,
                  "thresh": thresh,
                  "lmin": lmin,
                "RP_unthresh": [entry.tolist() for entry in RP_unthresh],
                  "RP_thresh": [entry.tolist() for entry in RP_thresh]}

    # Get attribute names to check whether "eps" is in there
    attributes = inspect.getmembers(RP, lambda a: not (inspect.isroutine(a)))
    attribute_names = [a[0] for a in attributes]
    # Add radius ("eps")
    if "eps" in attribute_names:
        RQA_output["radius"] = RP.eps

    # Save dictionary to .json file
    with open(filepath_output, 'w', encoding='utf-8') as outfile:
        json.dump(RQA_output, outfile, sort_keys=True, indent=4, ensure_ascii=True)  # indent makes it easier to read

    return filepath_output


    # Return unthresholded recurrence matrix and radius necessary to obtain fixed recurrence rate of thresh



from scipy.spatial import distance_matrix
import scipy


class class_RP:

    def __init__(self, x, method, thresh, theiler_window, compute_rp=True, **kwargs):
        """
        Class RP for computing recurrence plots from univariate time series.
        The recurrence_plot class supports time-delay embedding of multi-dimensional time series with
        known embedding dimension and delay, computation of euclidean distance matrices and
        computation of recurrence matrices based on four common threshold selection criteria.
        Note that this is a very sparse implementation with limited functionality and more
        comprehensive implementations (incl. embedding, parameter estimation, RQA...) can be found elsewhere.
        For a comprehensive summary of the recurrence plot method refer to [Marwan2007].

        If given a univariate/scalar time series, embedding parameters may be specified to apply
        time-delay embedding. If the time series multi-dimensional (non-scalar), no embedding
        can be applied and the input is treated as an embedded time series.
        If a recurrence plot should be given as input, 'compute_rp' has to be set to False.

    Parameters
    ----------
        x: 2D array (time, dimension)
            The time series to be analyzed, can be scalar or multi-dimensional.
        dim : int, optional
            embedding dimension (>1)
        tau : int, optional
            embedding delay
        method : str
             estimation method for the vicinity threshold
             epsilon (`distance`, `frr`, `stdev`, `fan`)
        thresh: float
            threshold parameter for the vicinity threshold,
            depends on the specified method (`epsilon`,
            `recurrence rate`, `multiple of standard deviation`,
            `fixed fraction of neighbours`)


    Examples
    --------

    - Create an instance of RP with fixed recurrence rate of 10%
      and without embedding:
           # >>> import RECLAC.recurrence_plot as rec
           # >>> RP(x=time_series, method='frr', thresh=0.1)

    - Create an instance of RP with a fixed recurrence threshold
      in units of the time series standard deviation (2*stdev) and with embedding:
           # >>> RP(x=time_series, dim=2, tau=3, method='stdev', thresh=2)

    - Obtain the recurrence matrix:

           # >>> a_rm = RP(x=time_series, dim=2, tau=3, method='stdev', thresh=2).rm()
    """

        #  Store time series as float
        self.x = x.copy().astype("float32")
        self.method = method
        self.thresh = thresh
        self.eps = np.nan  # Add radius to self
        self.theiler_window = theiler_window

        if compute_rp:
            #  Apply time-delay embedding: get embedding dimension and delay from **kwargs
            self.dim = kwargs.get("dim")
            self.tau = kwargs.get("tau")
            if self.dim is not None and self.tau is not None:
                assert (self.dim > 0) and (self.tau > 0), "Negative embedding parameter(s)!"
                #  Embed the time series
                self.embedding = self.embed()
            elif (self.dim is not None and self.tau is None) or (self.dim is None and self.tau is not None):
                raise NameError("Please specify either both or no embedding parameters.")
            else:
                if x.ndim > 1:
                    self.embedding = self.x
                else:
                    self.embedding = self.x.reshape(x.size, 1)

            # default metric: euclidean
            self.metric = kwargs.get("metric")
            if self.metric is None:
                self.metric = 'euclidean'
            assert (type(self.metric) is str), "'metric' must specified as string!"

            # Set threshold based on one of the four given methods (distance, fixed rr, fixed stdev, fan)
            # and compute recurrence matrix:
            self.R = self.apply_threshold()
            # 'back-up' (private) variable for self.counts to restore value when self.counts is altered
            self._R = np.copy(self.R)
        # RP is passed as x argument
        else:
            assert (x.ndim == 2), "If a recurrence matrix is provided, it has to be a 2-dimensional array."
            assert ~np.all(np.isnan(x)), "Recurrence matrix only contains NaNs."
            self.R = x

    def apply_threshold(self):
        """
        Apply thresholding to the distance matrix by one of four methods:

        *  'distance': no method, expects value for vicinity threshold

        *  'frr': fixed recurrence rate, expects specification of distance-distr. quantile (.xx)

        *  'stdev': standard deviation of time series, expects multiple of standard deviation

        *  'fan': fixed amounbt of neighbors, expects fraction of fixed neighbors

    Returns
    -------

        R: 2D array (integer)
            recurrence matrix
        """
        # compute distance matrix
        dist = class_RP.dist_mat(self, metric=self.metric, theiler_window=self.theiler_window)
        # initialize recurrence matrix
        a_rm = np.zeros(dist.shape, dtype="int")
        # different methods
        method = self.method
        thresh = self.thresh
        if method == "distance":
            i = np.where(dist <= thresh)
            a_rm[i] = 1
        elif method == "frr":
            eps = np.nanquantile(dist, thresh)
            self.eps = eps # Add radius to self
            i = np.where(dist <= eps)
            a_rm[i] = 1
        elif method == "stdev":
            eps = thresh * np.nanstd(self.x)
            self.eps = eps # Add radius to self
            i = np.where(dist <= eps)
            a_rm[i] = 1
        elif method == "fan":
            nk = np.ceil(thresh * a_rm.shape[0]).astype("int")
            i = (np.arange(a_rm.shape[0]), np.argsort(dist, axis=0)[:nk])
            a_rm[i] = 1
        else:
            raise NameError("'method' must be one of 'distance', 'frr', 'stdev' or 'fan'.")
        return a_rm

    def dist_mat(self, metric, theiler_window):
        """
        Returns a square-distance matrix with some specified metric.
        The following metrics can be used:
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean',
        'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'

    Parameters
    ----------
        metric: str
            Metric that is used for distance computation.
        theiler_window : int
            Number of diagonals to exclude.

    Returns
    -------

        R: 2D array (float)
            distance matrix
        """
        z = self.embedding
        # z= copy.deepcopy(data)

        # using the scipy.spatial implementation:
        if z.ndim == 1:
            z = z.reshape((z.size, 1))
        a_dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(z, metric=metric), force='tomatrix')
        # Exclude entries that fall within Theiler window

        # Create flattened list of row and column indices for each diagonal
        N = a_dist.shape[0]
        rows, cols = np.indices((N,N))
        row_vals = [item for sublist in [np.diag(rows, k=-i).tolist() for i in range(1, theiler_window+1)] for item in sublist]
        col_vals = [item for sublist in [np.diag(cols, k=-i).tolist() for i in range(1, theiler_window+1)] for item in sublist]
        # Set entries within Theiler window to NaN
        a_dist[row_vals, col_vals] = np.nan

        # Make matrix symmetric
        a_dist_theiler = np.tril(a_dist) + np.triu(a_dist.T, 1)
        # tril(m, k=0) gets the lower triangle of a matrix m (returns a copy of the matrix m with all elements above the kth diagonal zeroed). Similarly, triu(m, k=0) gets the upper triangle of a matrix m (all elements below the kth diagonal zeroed).
        # To prevent the diagonal being added twice, one must exclude the diagonal from one of the triangles, using either np.tril(A) + np.triu(A.T, 1) or np.tril(A, -1) + np.triu(A.T).
        if np.all(np.isnan(a_dist_theiler)):
            raise ValueError("Distance matrix only contains NaNs.")
        return a_dist_theiler

    def embed(self):
        """
        Time-delay embedding: embeds a scalar time series 'x' in 'dim' dimensions with time delay 'tau'.
    Returns
    -------

        R: 2D array (float)
             embedded time series
        """
        K = (self.dim - 1) * self.tau
        assert (K < self.x.size), "Choice of embedding parameters exceeds time series length."
        # embedd time series:
        a_emb = np.asarray([np.roll(self.x, -d * self.tau)[:-K] for d in range(self.dim)]).T
        return a_emb

    def rm(self):
        """
        Returns the (square) recurrence matrix.
    Returns
    -------

        R: 2D array (int)
             recurrence matrix
        """
        return self.R

    def line_hist(self, linetype, border=None):
        """
        Extracts all diagonal/vertical lines from a recurrence matrix. The 'linetype'
        arguments specifies which lines should be analysed. Returns all identified
        line lengths and the line length histogram.
        Since the length of border lines is generally unknown, they can be discarded or
        replaced by the mean/max line length.

    Parameters
    ----------
        linetype: str
            specifies whether diagonal ('diag') or vertical ('vert') lines
            should be extracted
        border: str, optional
            treatment of border lines: None, 'discard', 'kelo', 'mean' or 'max'


    Returns
    -------

        a_ll, a_bins, a_lhist: tuple of three 1D float arrays
            line lengths, bins, histogram

    Examples
    --------

    - Create an instance of RP with fixed recurrence rate of 10% for a noisy
      sinusoidal and obtain the diagonal line length histogram without border lines:
        # >>> import RECLAC.recurrence_plot as rec
        # >>> # sinusoidal with five periods and superimposed white Gaussian noise
        # >>> a_ts = np.sin(2*math.pi*np.arange(100)/20) + np.random.normal(0, .25, 100)
        # >>> # define a recurrence plot instance with a fixed recurrence rate of 10%
        # >>> RP = rec.RP(a_ts, method='frr', dim=2, per=5, thresh=.1)
        # >>> # obtain line histogram for diagonal lines while border lines are discarded
        # >>> _, a_bins, a_freqs = RP.line_hist(linetype='diag', border='discard')
        # >>> a_bins, a_freqs
        (array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]), array([152,  60,  16,  11,   1]))

    - Obtain the vertical line length histogram with no action on border lines:
        # >>> _, a_vbins, a_vfreqs = RP.line_hist(linetype='vert', border=None)
        # >>> a_vbins, a_vfreqs
        (array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]), array([268, 158,  71,  18,   5]))
        """
        N = self.R.shape[0]
        a_ll = np.array([])
        # 'counter' counts border lines
        counter = 0
        for n in range(1, N):
            # grab the n-th diagonal
            tmp_line = self._extract_line(n, linetype)
            # run length encoding
            tmp_rle = class_RP._rle(tmp_line)

            ## Border lines
            if border is not None:
                if tmp_rle[0][0] == 1:
                    tmp_rle = tmp_rle[1:, ]
                    counter += 1
                try:
                    if tmp_rle[-1][0] == 1:
                        tmp_rle = tmp_rle[:-1, ]
                        counter += 1
                except IndexError:
                    tmp_rle = tmp_rle

            ## Find diagonal lines
            tmp_ind = np.where(tmp_rle[:, 0] == 1)
            # collect their lengths
            tmp_lengths = tmp_rle[tmp_ind, 1].ravel()
            if tmp_lengths.size > 0:
                a_ll = np.hstack([a_ll, tmp_lengths])
                ## Append border line substitutes (if desired)
                if border == 'mean':
                    avgll = np.mean(a_ll)
                    a_ll = np.hstack([a_ll, np.repeat(avgll, counter)])
                elif border == 'max':
                    maxll = np.max(a_ll)
                    a_ll = np.hstack([a_ll, np.repeat(maxll, counter)])
                elif border == 'kelo':
                    maxll = np.max(a_ll)
                    a_ll = np.hstack([a_ll, maxll])

        # any lines?
        if a_ll.size > 0:
            a_bins = np.arange(0.5, np.max(a_ll) + 0.1 + 1, 1.)
            a_lhist, _ = np.histogram(a_ll, bins=a_bins)
            return a_ll, a_bins, a_lhist
        else:
            raise ValueError("No lines could be identified.")
            return None

    def _extract_line(self, n, linetype):
        """
        Extracts the n-th diagonal/column from a recurrence matrix, depending on
        whether diagonal (linetype='diag') or vertical (linetype='vert') lines are
        desired.

    Parameters
    ----------
        n: int
            index of diagonal/column of the RP (0 corresponds to LOI for diagonals)
        linetype: str
            specifies whether diagonal ('diag') or vertical ('vert') lines
            should be extracted

    Returns
    -------

        1D float array
            n-th diagonal/column of recurrence matrix
        """
        if linetype == 'diag':
            return np.diag(self.R, n)
        elif linetype == 'vert':
            return self.R[:, n]
        else:
            print("Specification error: 'linetype' must be one of 'diag' or 'vert'.")

    @staticmethod
    def _rle(sequence):
        """
        Run length encoding: count consecutive occurences of values in a sequence.
        Applied to binary sequences (diagonals/columns of recurrence matrix) to obtain
        line lengths.

    Parameters
    ----------
        sequence: 1D float array
            sequence of values (0s and 1s for recurrence plots)
    Returns
    -------

        2D float array
            run values (first column) and run lengths (second column)
        """
        # src:  https://github.com/alimanfoo
        ## Run length encoding: Find runs of consecutive items in an array.

        # ensure array
        x = np.asanyarray(sequence)
        if x.ndim != 1:
            raise ValueError('only 1D array supported')
        n = x.shape[0]

        # handle empty array
        if n == 0:
            return np.array([]), np.array([]), np.array([])

        else:
            # find run starts
            loc_run_start = np.empty(n, dtype=bool)
            loc_run_start[0] = True
            np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
            run_starts = np.nonzero(loc_run_start)[0]

            # find run values
            run_values = x[loc_run_start]

            # find run lengths
            run_lengths = np.diff(np.append(run_starts, n))

            # stack and return
            return np.vstack([run_values, run_lengths]).T

    @staticmethod
    def _fraction(bins, hist, lmin):
        """
        Returns the fraction of lines that are longer than 'lmin' based on the line
        length histogram. For diagonal (vertical) lines, this corresponds to DET (LAM).

    Parameters
    ----------
        bins: 1D float array
            bins of line length histogram
        hist: 1D float array
            frequencies of line lengths within each bin
        lmin: int value
             minimum line length
    Returns
    -------

        rq: float value
            fraction of diagonal/vertical lines that exceed 'lmin' (DET/LAM)
        """
        # find fraction of lines larger than lmin
        a_Pl = hist.astype('float')
        a_l = (0.5 * (bins[:-1] + bins[1:])).astype('int')
        ind = a_l >= lmin
        # compute fraction
        a_ll_large = a_l[ind] * a_Pl[ind]
        a_ll_all = a_l * a_Pl
        rq = a_ll_large.sum() / a_ll_all.sum()
        return rq

    def RQA(self, lmin, measures='all', border=None):
        """
        Performs a recurrence quantification analysis (RQA) on a recurrence matrix based on
        a list of (traditional) recurrence quantification measures. Returns the following
        nine measures by default:
        - recurrence rate
        - DET
        - average diagonal line length
        - maximum diagonal line length
        - LAM
        - average vertical line length
        - maximum vertical line length
        - average white vertical line length(/recurrence time)
        - maximum white vertical line length(/recurrence time)

        If only quantifiers based on diagonal/vertical/white lines are desired, this can
        be restricted with the 'measure' argument.


    Parameters
    ----------
        lmin: int value
             minimum line length
        measures: str
            determines which recurrence quantification measures are computed
            ('all', 'diag', 'vert', 'white')
        border: str
            treatment of border lines: None, 'discard', 'mean' or 'max'


    Returns
    -------

        d_rqa: float dictionary
            recurrence quantification measures

    Examples
    --------

    - Create an instance of RP with fixed recurrence rate of 10% for a noisy
      sinusoidal and run a full recurrence quantification analysis:
        # >>> import RECLAC.recurrence_plot as rec
        # >>> # sinusoidal with five periods and superimposed white Gaussian noise
        # >>> a_ts = np.sin(2*math.pi*np.arange(100)/20) + np.random.normal(0, .25, 100)
        # >>> # define a recurrence plot instance with a fixed recurrence rate of 10%
        # >>> RP = rec.RP(a_ts, method='frr', thresh=.1)
        # >>> # compute all RQA measures with no border correction:
        # >>> RP.RQA(lmin=2, measures='all', border=None)
        {'RR': 0.10005540166204986,
        'DET': 0.6064356435643564,'avgDL': 2.5257731958762886,'maxDL': 9,
         'LAM': 0.7002237136465325,'avgVL': 2.484126984126984,'maxVL': 5,
         'avgWVL': 15.614931237721022,'maxWVL': 86}
    - Run only a recurrence quantification analysis that considers diagonal measures
      on lines of minimum length 3 whereas border lines are set to the average diagonal
      line length:
        # >>> RP.RQA(lmin=3, measures='diag', border='mean')
        {'RR': 0.10005540166204986,
         'DET': 0.12262958280657396,'avgDL': 3.4642857142857144,'maxDL': 5,
         'LAM': None, 'avgVL': None, 'maxVL': None,
         'avgWVL': None, 'maxWVL': None}
        """
        DET, avgDL, maxDL, N_DL, ENT_DL, LAM, avgVL, maxVL, N_VL, ENT_VL, avgWL, maxWL = np.repeat(None, 12)
        # recurrence rate
        rr = np.sum(self.R) / (self.R.size)
        # diagonal line structures
        if (measures == 'diag') or (measures == 'all'):
            # Compute histogram of diagonal lines
            a_ll, a_bins, a_nlines = self.line_hist(linetype='diag', border=border)
            DET = class_RP._fraction(bins=a_bins, hist=a_nlines, lmin=lmin)
            a_llsub = a_ll[a_ll >= lmin]  # I think all returns a vector of line lengths for each line
            avgDL = np.mean(a_llsub)  # Average diagonal line length
            maxDL = np.max(a_llsub).astype(int)  # Maximum diagonal line length
            nDL = len(a_llsub)  # Number of diagonal lines >= minimum line length
            a_nlines_geq_lmin = a_nlines[(a_bins >= lmin)[
                                         1:]]  # Get number of lines in each bin without including lengths < lmin; select [1:] because the bin edges are of length len(a_nlines) + 1
            ENT_DL = scipy.stats.entropy(a_nlines_geq_lmin)  # Diagonal line distribution entropy

        # vertical line structures
        if (measures == 'vert') or (measures == 'all'):
            # Compute histogram of vertical lines
            a_ll, a_bins, a_nlines = self.line_hist(linetype='vert', border=border)
            LAM = class_RP._fraction(bins=a_bins, hist=a_nlines, lmin=lmin)
            a_llsub = a_ll[a_ll >= lmin]  # Select lines >= minimum line length
            avgVL = np.mean(a_llsub)  # Average diagonal line length
            maxVL = np.max(a_llsub).astype(int)  # Maximum diagonal line length
            nVL = len(a_llsub)  # Number of vertical lines >= minimum line length
            a_nlines_geq_lmin = a_nlines[(a_bins >= lmin)[
                                         1:]]  # Get number of lines in each bin without including lengths < lmin; select [1:] because the bin edges are of length len(a_nlines) + 1
            ENT_VL = scipy.stats.entropy(a_nlines_geq_lmin)  # Vertical line distribution entropy

        # white vertical line structures/ recurrence times
        if (measures == 'white') or (measures == 'all'):
            self.R = 1 - self.R
            a_ll, a_bins, a_nlines = self.line_hist(linetype='vert', border=border)
            a_llsub = a_ll[a_ll >= lmin]
            avgWL = np.mean(a_llsub)
            maxWL = np.max(a_llsub).astype(int)
            # restore value
            self.R = self._R

        d_rqa = dict([('RR', rr),
                      ('DET', DET), ('avgDL', avgDL), ('maxDL', maxDL), ('nDL', nDL), ('ENT_DL', ENT_DL),
                      ('LAM', LAM), ('avgVL', avgVL), ('maxVL', maxVL), ('nVL', nVL), ('ENT_VL', ENT_VL),
                      ('avgWVL', avgWL), ('maxWVL', maxWL)])

        return d_rqa


# from pyrqa.settings import Settings
# >>> from pyrqa.neighbourhood import FixedRadius
# >>> from pyrqa.metric import EuclideanMetric
# >>> from pyrqa.computation import RQAComputation
# >>> time_series = [0.1, 0.5, 0.3, 1.7, 0.8, 2.4, 0.6, 1.2, 1.4, 2.1, 0.8]
# >>> settings = Settings(time_series,
#                         embedding_dimension=3,
#                         time_delay=1,
#                         neighbourhood=FixedRadius(1.0),
#                         similarity_measure=EuclideanMetric,
#                         theiler_corrector=1,
#                         min_diagonal_line_length=2,
#                         min_vertical_line_length=2,
#                         min_white_vertical_line_length=2)
# >>> computation = RQAComputation.create(settings, verbose=True)
# >>> result = computation.run()
# >>> print result
# Recurrence plot computations can be conducted likewise. Building on the previous example::
#
# >>> from pyrqa.computation import RecurrencePlotComputation
# >>> from pyrqa.image_generator import ImageGenerator
# >>> computation = RecurrencePlotComputation.create(settings)
# >>> result = computation.run()
# >>> ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse, 'recurrence_plot.png')