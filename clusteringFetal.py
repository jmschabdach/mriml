import clusteringLib as cl

import numpy as np
import pandas as pd
from scipy import stats
import os

def loadData(sitePath):
    boldFdFn = os.path.join("metrics", "BOLD-displacement-metrics.csv")
    boldDvarsFn = os.path.join("metrics", "BOLD-intensity-metrics.csv")
    boldCrMatFn = os.path.join("metrics", "original-correlation-matrix.csv")
    boldDiceMatFn = os.path.join("metrics", "BOLD_dice_mat.csv")
    boldMiMatFn = os.path.join("metrics", "BOLD_mi_mat.csv")

    dagFdFn = os.path.join("metrics", "corrected_dag-displacement-metrics.csv")
    dagDvarsFn = os.path.join("metrics", "corrected_dag-intensity-metrics.csv")
    dagCrMatFn = os.path.join("metrics", "dag-correlation-matrix.csv")
    dagDiceMatFn = os.path.join("metrics", "corrected_dag_dice_mat.csv")
    dagMiMatFn = os.path.join("metrics", "corrected_dag_mi_mat.csv")
    
    tradFdFn = os.path.join("metrics", "corrected_traditional-displacement-metrics.csv")
    tradDvarsFn = os.path.join("metrics", "corrected_traditional-intensity-metrics.csv")
    tradCrMatFn = os.path.join("metrics", "traditional-correlation-matrix.csv")
    tradDiceMatFn = os.path.join("metrics", "corrected_traditional_dice_mat.csv")
    tradMiMatFn = os.path.join("metrics", "corrected_traditional_mi_mat.csv")
    pBoldFdDf = pd.DataFrame()
    pBoldDvarsDf = pd.DataFrame()
    pDagFdDf = pd.DataFrame()
    pDagDvarsDf = pd.DataFrame()
    pTradFdDf = pd.DataFrame()
    pTradDvarsDf = pd.DataFrame()
    
    pBoldDiceDf = pd.DataFrame()
    pBoldMiDf = pd.DataFrame()
    pTradDiceDf = pd.DataFrame()
    pTradMiDf = pd.DataFrame()
    pDagDiceDf = pd.DataFrame()
    pDagMiDf = pd.DataFrame()
    
    subjs = sorted(os.listdir(sitePath))
    for subj in subjs:
        print(subj)
        subjPath = os.path.join(sitePath, subj)
        if os.path.isdir(subjPath):
            # load the BOLD FD data
            df = pd.read_csv(os.path.join(subjPath, boldFdFn))[:100]
            df = df.rename(columns={"0":subj})
            # add the BOLD FD data to the dataframe of BOLD FD data
            pBoldFdDf = pd.concat([pBoldFdDf, df], axis=1)
    
            # load the BOLD DVARS data
            df2 = pd.read_csv(os.path.join(subjPath, boldDvarsFn))[:100]
            df2 = df2.rename(columns={"0":subj})
            # add the BOLD FD data to the dataframe of BOLD FD data
            pBoldDvarsDf = pd.concat([pBoldDvarsDf, df2], axis=1)
    
            # load the DAG FD data
            df = pd.read_csv(os.path.join(subjPath, dagFdFn))[:100]
            df = df.rename(columns={"0":subj})
            # add the BOLD FD data to the dataframe of BOLD FD data
            pDagFdDf = pd.concat([pDagFdDf, df], axis=1)
    
            # load the DAG DVARS data
            df2 = pd.read_csv(os.path.join(subjPath, dagDvarsFn))[:100]
            df2 = df2.rename(columns={"0":subj})
            # add the BOLD FD data to the dataframe of BOLD FD data
            pDagDvarsDf = pd.concat([pDagDvarsDf, df2], axis=1)
    
            # load the trad FD data
            df = pd.read_csv(os.path.join(subjPath, tradFdFn))[:100]
            df = df.rename(columns={"0":subj})
            # add the BOLD FD data to the dataframe of BOLD FD data
            pTradFdDf = pd.concat([pTradFdDf, df], axis=1)
    
            # load the trad DVARS data
            df2 = pd.read_csv(os.path.join(subjPath, tradDvarsFn))[:100]
            df2 = df2.rename(columns={"0":subj})
            # add the BOLD FD data to the dataframe of BOLD FD data
            pTradDvarsDf = pd.concat([pTradDvarsDf, df2], axis=1)
    
            # BOLD Dice
            mat = np.loadtxt(open(os.path.join(subjPath, boldDiceMatFn), "r"), delimiter=",")[:100, :100]
            df = pd.DataFrame(data=mat.flatten().T, columns=[subj])
            pBoldDiceDf = pd.concat([pBoldDiceDf, df], axis=1)
    
            # BOLD MI
            mat = np.loadtxt(open(os.path.join(subjPath, boldMiMatFn), "r"), delimiter=",")[:100, :100]
            df = pd.DataFrame(data=mat.flatten().T, columns=[subj])
            pBoldMiDf = pd.concat([pBoldMiDf, df], axis=1)

            # DAG Dice
            mat = np.loadtxt(open(os.path.join(subjPath, dagDiceMatFn), "r"), delimiter=",")[:100, :100]
            df = pd.DataFrame(data=mat.flatten().T, columns=[subj])
            pDagDiceDf = pd.concat([pDagDiceDf, df], axis=1)

            # DAG MI
            mat = np.loadtxt(open(os.path.join(subjPath, dagMiMatFn), "r"), delimiter=",")[:100, :100]
            df = pd.DataFrame(data=mat.flatten().T, columns=[subj])
            pDagMiDf = pd.concat([pDagMiDf, df], axis=1)
    
            # Traditional Dice
            mat = np.loadtxt(open(os.path.join(subjPath, tradDiceMatFn), "r"), delimiter=",")[:100, :100]
            df = pd.DataFrame(data=mat.flatten().T, columns=[subj])
            pTradDiceDf = pd.concat([pTradDiceDf, df], axis=1)
    
            # Traditional MI
            mat = np.loadtxt(open(os.path.join(subjPath, tradMiMatFn), "r"), delimiter=",")[:100, :100]
            df = pd.DataFrame(data=mat.flatten().T, columns=[subj])
            pTradMiDf = pd.concat([pTradMiDf, df], axis=1)

    # Drop rows containing nan
    boldFdDf = pBoldFdDf.fillna(2)
    boldDvarsDf = pBoldDvarsDf.fillna(50)
    boldDiceDf = pBoldDiceDf.fillna(0)
    boldMiDf = pBoldMiDf.fillna(0)
    
    tradFdDf = pTradFdDf.fillna(2)
    tradDvarsDf = pTradDvarsDf.fillna(50)
    tradDiceDf = pTradDiceDf.fillna(0)
    tradMiDf = pTradMiDf.fillna(0)
    
    dagFdDf = pDagFdDf.fillna(2)
    dagDvarsDf = pDagDvarsDf.fillna(50)
    dagDiceDf = pDagDiceDf.fillna(0)
    dagMiDf = pDagMiDf.fillna(0)
    
    boldFdDf = boldFdDf.T.drop_duplicates().T
    boldDvarsDf = boldDvarsDf.T.drop_duplicates().T
    boldDiceDf = boldDiceDf.T.drop_duplicates().T
    boldMiDf = boldMiDf.T.drop_duplicates().T
    
    tradFdDf = tradFdDf.T.drop_duplicates().T
    tradDvarsDf = tradDvarsDf.T.drop_duplicates().T
    tradDiceDf = tradDiceDf.T.drop_duplicates().T
    tradMiDf = tradMiDf.T.drop_duplicates().T
    
    dagFdDf = dagFdDf.T.drop_duplicates().T
    dagDvarsDf = dagDvarsDf.T.drop_duplicates().T
    dagDiceDf = dagDiceDf.T.drop_duplicates().T
    dagMiDf = dagMiDf.T.drop_duplicates().T

    return boldFdDf, dagFdDf, tradFdDf, boldDvarsDf, dagDvarsDf, tradDvarsDf, boldDiceDf, dagDiceDf, tradDiceDf, boldMiDf, dagMiDf, tradMiDf

def loadMeta(demoFn):
    demoDf = pd.read_csv(demoFn)

    # Convert MF to Male Female
    if 'Fem=0 Male=1' in list(demoDf):
        demoDf = demoDf.rename(columns={'Fem=0 Male=1': 'Sex'})
    
    demoDf['Age_Group'] = ["Preadolescent" for i in range(demoDf.shape[0])]
    
    demoDf['Sex/Age'] = demoDf['Sex'] + ' ' + demoDf['Age At Scan'].astype(str)
    demoDf['Sex/Cohort'] = demoDf['Sex'] + ' ' + demoDf['Cohort']
    demoDf['Age/Cohort'] = demoDf['Cohort'] + ' ' + demoDf['Age At Scan'].astype(str)
    
    if "Unnamed: 0" in list(demoDf):
        demoDf = demoDf.drop(columns=["Unnamed: 0"])

    return demoDf

def unionDfs(df, demoDf):
    # Filter DFs    
    print(len(list(demoDf['ID'])))
    print(len(list(df)))

    metaScans = list(set(list(demoDf['ID'])))
    metricScans = list(set(list(df)))

    print(len(metaScans))
    print(len(metricScans))

    extraMeta = [ i for i in metaScans if i not in metricScans]
    extraMetrics = [ i for i in metricScans if i not in metaScans]

    metaScans = [i for i in metaScans if i not in extraMeta]

    demoDf = demoDf[demoDf['ID'].isin(metaScans)]
    df = df.drop(columns=extraMetrics)

    demoDf = demoDf.drop_duplicates(subset=['ID'])
    
    return df, demoDf

def reduceAndCluster(labels, data, demo, k, ctype):
    idxDrop = []
    vals, counts = np.unique(labels, return_counts=True)
    for i, j in zip(vals, counts):
        if j < 0.1*data.shape[1]:
            idxDrop.extend(np.where(labels == i)[0])
            
    idxDrop = [int(i) for i in idxDrop]
    colDrop = [list(data)[i] for i in idxDrop]
    
    # Drop outlier values from the dataframe
    dfFewer = data.drop(columns=colDrop)
    dfFewer, demoFewer = unionDfs(dfFewer, demo)

    if dfFewer.shape[1] > k :
        if ctype == "spectral":
            try:
                slabels2 = cl.spectralClustering(dfFewer.T, k).labels_
            except:
                print("some error happened")                  
                slabels2 = [0]
        elif ctype == "kmeans":
            slabels2 = cl.kmeansClustering(dfFewer.T, k).labels_
        elif ctype == "agg":
            slabels2 = cl.sklearnAgg(dfFewer.T, k).labels_
    else:
        slabels2 = [0]

    vals, counts = np.unique(slabels2, return_counts=True)
    for i, j in zip(vals, counts):
        print(i, j)
        
    demoFewer = demoFewer.reset_index()
    demoFewer = demoFewer.drop(columns=["index"])

    dfFewer = dfFewer.reset_index()
    dfFewer = dfFewer.drop(columns=["index"])
        
    return slabels2, dfFewer, demoFewer

def main():
    base = "/home/jms565/Research/mriml/"
    ctype = "agg" # problems with spectral for preads
    agegroup = "fetal"
    subjDir = "/mnt/research/data/Fetal/brain/"
    outFn = base+"clustering_results/"+agegroup+"_"+ctype+".csv"
    demoFn = base+"demographics/cleaned_"+agegroup+"_scan_info.csv"

    boldFd, dagFd, tradFd, boldDvars, dagDvars, tradDvars, boldDice, dagDice, tradDice, boldMi, dagMi, tradMi = loadData(subjDir)

    dfDemo = loadMeta(demoFn)

    # Calculate the differences between the original and registered metrics
    deltaTradFd = boldFd - tradFd
    deltaTradDvars = boldDvars - tradDvars
    deltaTradDice = boldDice - tradDice
    deltaTradMi = boldMi - tradMi

    deltaDagFd = boldFd - dagFd
    deltaDagDvars = boldDvars - dagDvars
    deltaDagDice = boldDice - dagDice
    deltaDagMi = boldMi - dagMi

    deltaTradFd = deltaTradFd.fillna(2)
    deltaTradDvars = deltaTradDvars.fillna(50)
    deltaTradDice = deltaTradDice.fillna(0)
    deltaTradMi = deltaTradMi.fillna(0)

    deltaDagFd = deltaDagFd.fillna(2)
    deltaDagDvars = deltaDagDvars.fillna(50)
    deltaDagDice = deltaDagDice.fillna(0)
    deltaDagMi = deltaDagMi.fillna(0)

    metrics = [boldFd, boldDvars, boldDice, boldMi,
               deltaTradFd, deltaTradDvars, deltaTradDice, deltaTradMi,
               deltaDagFd, deltaDagDvars, deltaDagDice, deltaDagMi]

    metricsText = ["BOLD FD", "BOLD DVARS", "BOLD Dice", "BOLD MI",
                   "dTrad FD", "dTrad DVARS", "dTrad Dice", "dTrad MI",
                   "dDag FD", "dDag DVARS", "dDag Dice", "dDag MI"]
    
    demo = ['Cohort', 'Sex', 'Age At Scan', 'Sex/Age','Sex/Cohort','Age/Cohort']
    
    for d in demo: 
        with open(outFn, 'a') as f:
            f.write("\n")
            f.write("Demographic Label, "+str(d)+"\n")
        
        for m, n in zip(metrics, metricsText):
            clusters = len(np.unique(dfDemo[d].get_values()))
            mDf, dlabels = unionDfs(m, dfDemo)
            origSize = mDf.shape[1]
   
            labels = cl.kmeansClustering(mDf.T, clusters).labels_
            labelCounts = [np.count_nonzero(labels == i) for i in np.unique(labels)]
    
            while min(labelCounts) < 0.1*len(labels) and mDf.shape[1] > 0.3*origSize and len(labelCounts) > 1:
                labels, mDf, dlabels = reduceAndCluster(labels, mDf, dlabels, clusters, ctype)
                print(mDf.shape)
                labelCounts = [ np.count_nonzero(labels == i) for i in np.unique(labels)]
        
            if mDf.shape[1] > 0.3*origSize and len(labelCounts) > 1:
                print(ctype, n)
                cl.analyzeClusterContents(dlabels[d].get_values(), labels, outFn)

if __name__ == "__main__":
    main()
