import pickle
from ROOT import TFile
from root_pandas import read_root

def addbdtscore(infile,tree):
    ifile = open("discriminator_resolved.pickle")
    model = pickle.load(ifile)

    vars_to_load_ = ['MET','METSig','Jet1Pt', 'Jet1Eta', 'Jet1Phi','Jet2Pt', 'Jet2Eta', 'Jet2Phi','DiJetMass','DiJetPt', 'DiJetEta','DiJetPhi','nJets','met_Phi']
    
    if not ("SR" in tree or "SBand" in tree):vars_to_load_[0]="RECOIL"
    df = read_root(infile,tree,columns=vars_to_load_)
    #df=df[vars_to_load_]
    print df[:1]
    out=model.decision_function(df).ravel()
    
    print out[:10]
    return out
