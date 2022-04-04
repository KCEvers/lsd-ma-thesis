import pathlib
import json
import bids
import numpy as np
import nibabel as nib
import math
import time
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.signal
import statsmodels.regression.rolling
import EntropyHub as EH
import nolitsa
import nolitsa.utils
import nolitsa.data
# from nolitsa import data, lyapunov, utils, d2
import nolds as nolds
import ordpy
import warnings
import nilearn
import copy
import pandas as pd
import re
import inspect
import subprocess
import functools
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import glob
import random
from natsort import natsorted
from PIL import Image
from nilearn import datasets, plotting
from nilearn.plotting import plot_epi
from nilearn.image import mean_img
import nitime
import datetime
import shutil
# from nitime.timeseries import TimeSeries  # Import the time-series objects
# from nitime.analysis import SpectralAnalyzer, FilterAnalyzer  # Import the analysis objects
# import nitime.fmri.io as io
import io
import toolz
import sklearn
import markdown
import os
import webbrowser
import copy
import chart_studio
import plotly.io as pio
import chart_studio.plotly as py
# import dplython
import nolitsa
import statsmodels.tsa.stattools
import arch.unitroot

# from statsmodels.tsa.stattools import adfuller # ADF test for stationarity
# from statsmodels.tsa.stattools import kpss # KPSS test for stationarity

from thesis.master import vars
