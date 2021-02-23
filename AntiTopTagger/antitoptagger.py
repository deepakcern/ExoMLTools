## following the instructions from https://betatim.github.io/posts/sklearn-for-TMVA-users/
## https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py

import random

import pandas as pd
import numpy as np
import matplotlib as mpl
#import matplotlib
mpl.use('pdf')
import matplotlib.pyplot as plt

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


def signal_background(data1, data2, columnName="", grid=True,
                      xlabel="", xlog=False, ylabel="",
                      ylog=False, 
                      sharey=False, figsize=None,
                      layout=None, bins=10, **kwds):
    
    
    ## check list of var is not empty 

    ## get the var to be plotted 
    
    '''
    if column is not None:
        if not isinstance(column, (list, np.ndarray, Index)):
            column = [column]
        data1 = data1[column]
        data2 = data2[column]
        
    data1 = data1._get_numeric_data()
    data2 = data2._get_numeric_data()
    naxes = len(data1.columns)

    fig, axes = plotting._subplots(naxes=naxes, ax=ax, squeeze=False,
                                   sharex=sharex,
                                   sharey=sharey,
                                   figsize=figsize,
                                   layout=layout)
    _axes = plotting._flatten(axes)
    for i, col in enumerate(com._try_sort(data1.columns)):
        ax = _axes[i]
        low = min(data1[col].min(), data2[col].min())
        high = max(data1[col].max(), data2[col].max())
        ax.hist(data1[col].dropna().values,
                bins=bins, range=(low,high), **kwds)
        ax.hist(data2[col].dropna().values,
                bins=bins, range=(low,high), **kwds)
        ax.set_title(col)
        ax.grid(grid)

    plotting._set_ticks_props(axes, xlabelsize=xlabelsize, xrot=xrot,
                              ylabelsize=ylabelsize, yrot=yrot)
    fig.subplots_adjust(wspace=0.3, hspace=0.7)
    plt.savefig('plots/antitop_comparison+.pdf')

    return axes

    '''

def plot_ROC(bdt, X_train, y_train, type_="train"):
    import matplotlib.pyplot as plt

    from sklearn.metrics import roc_curve, auc
    decisions = bdt.decision_function(X_train)
    # Compute ROC curve and area under the curve
    fpr, tpr, thresholds = roc_curve(y_train, decisions)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()

    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))
    
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    #plt.show()
    plt.savefig('plots/antitop_roc_'+type_+'.pdf')
    plt.close(fig)    
    
def compare_train_test(clf, X_train, y_train, X_test, y_test,postfix="_2b", bins=25):
    import matplotlib.pyplot as plt
    
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]
        
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')
    
    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    
    
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    
    plt.savefig("plots/antitop_bdt"+postfix+".pdf")




''' load data '''
from root_pandas import read_root

vars_to_load_ = ["MET", "dPhi_jetMET", "Jet1Pt", "Jet1Eta", "Jet1deepCSV", "Jet2Pt", "Jet2Eta", "Jet2deepCSV", "nPV", "pfpatCaloMETPt","pfTRKMETPt","delta_pfCalo","rJet1PtMET","ratioPtJet21","dPhiJet12","dEtaJet12" ]#, "Jet3Pt", "Jet3Eta", "Jet3deepCSV","nPV","isjet2EtaMatch"]

signal_file_ = "merged_signal.root" #"signal_Ma250_MChi1_MA1200_tanb35_sint_0p7_MH_600_MHC_600.root"
bkg_file_    = "merged_backgroud.root"# "tt_semileptonic.root"

df_signal = read_root(signal_file_, 'bbDM_SR_2b', columns=vars_to_load_)

df_bkg    = read_root(bkg_file_,    'bbDM_SR_2b', columns=vars_to_load_)
print df_signal[:1]

''' skim the data '''
#df_signal_skim = df_signal[ (df_signal.M_Jet1AK8Jet > 0.) ]
#df_bkg_skim = df_bkg[(df_bkg.M_Jet1AK8Jet > 0.)]

df_signal_skim = df_signal
df_bkg_skim =    df_bkg

signal_background(df_signal_skim, df_bkg_skim, 
                  column=vars_to_load_,
                  bins=20)


print "size of the dataset", len(df_signal_skim), len(df_bkg_skim)
#print df_signal_skim
#print df_bkg_skim

# join signal and background sample into same dataset. 
X = np.concatenate((df_signal_skim, df_bkg_skim))
#print X

## create a column with length = sum of length of signal and background, signal is 1 and background is 0
y = np.concatenate((np.ones(df_signal_skim.shape[0]),
                    np.zeros(df_bkg_skim.shape[0])))



''' plot data 1d '''



''' plot data 2d / scatter '''

'''plot data correlation/covariance '''

''' split samples for testing and training ''' 


X_dev,X_eval, y_dev,y_eval = train_test_split(X, y,
                                              test_size=0.01, random_state=42)

X_train,X_test, y_train,y_test = train_test_split(X_dev, y_dev,
                                                  test_size=0.33, random_state=42)



''' define model '''
dt = DecisionTreeClassifier(max_depth=3,
                            min_samples_leaf=0.03)#*len(X_train))

bdt = AdaBoostClassifier(dt,
                         algorithm='SAMME',
                         n_estimators=600,
                         learning_rate=0.05)


''' perform training ''' 
bdt.fit(X_train, y_train)

print "training done "


y_predicted = bdt.predict(X_test)
print classification_report(y_test, y_predicted,
                            target_names=["background", "signal"])
print "Area under ROC curve X_test: %.4f"%(roc_auc_score(y_test,
                                                  bdt.decision_function(X_test)))





y_predicted = bdt.predict(X_train)
print classification_report(y_train, y_predicted,
                            target_names=["background", "signal"])
print "Area under ROC curve X_train: %.4f"%(roc_auc_score(y_train,
                                                  bdt.decision_function(X_train)))



y_predicted = bdt.predict(X_eval)
print classification_report(y_eval, y_predicted,
                            target_names=["background", "signal"])
print "Area under ROC curve X_eval: %.4f"%(roc_auc_score(y_eval,
                                                  bdt.decision_function(X_eval)))

''' check the outcome ''' 

''' check the performance / ROC ''' 





''' overtraining check '''

''' save trained model for future, i.e. applying to the analysis ''' 

''' going further: deeper ? ''' 
plot_ROC(bdt, X_train, y_train, "train_1b")
plot_ROC(bdt, X_test, y_test, "test_1b")
plot_ROC(bdt, X_eval, y_eval, "eval_1b")

compare_train_test(bdt, X_train, y_train, X_test, y_test,postfix="_1b")

from pickle import dump, load
dump(bdt, open('SR2b_discriminator_v0.pickle','wb')) 

