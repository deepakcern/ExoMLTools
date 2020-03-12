
## following the instructions from https://betatim.github.io/posts/sklearn-for-TMVA-users/
import random

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pandas.core.common as com
from pandas.core.index import Index

from pandas.tools import plotting
from pandas.tools.plotting import scatter_matrix

from sklearn.cross_validation import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.metrics import roc_curve, auc


''' load data '''
from root_pandas import read_root

vars_to_load_ = ["Jet1Eta","Jet1Pt","isAK4jet1EtaMatch","M_Jet1AK8Jet","min_dphi_jets","min_dPhi"]
signal_file_ = "/tmp/khurana/EXO-ggToXdXdHToBB_sinp_0p35_tanb_1p0_mXd_10_MH3_1200_MH4_150_MH2_1200_MHC_1200_CP3Tune_13TeV_0000_0.root"
bkg_file_    = "/tmp/khurana/Merged_TopSemileptonic.root"

df_signal = read_root(signal_file_, 'monoHbb_SR_boosted', columns=vars_to_load_)
df_bkg = read_root(signal_file_, 'monoHbb_SR_boosted', columns=vars_to_load_)


''' skim the data '''
df_signal_skim = df_signal[ (df_signal.M_Jet1AK8Jet > 0.) ]
df_bkg_skim = df_bkg[(df_bkg.M_Jet1AK8Jet > 0.)]

print df_signal_skim

print df_bkg_skim
''' plot data 1d '''



''' plot data 2d / scatter '''

'''plot data correlation/covariance '''

''' split samples for testing and training ''' 

''' define model '''

''' perform training ''' 

''' check the outcome ''' 

''' check the performance / ROC ''' 

''' overtraining check '''

''' save trained model for future, i.e. applying to the analysis ''' 

''' going further: deeper ? ''' 
