import pandas as pd
import numpy as np
import os

## Load a vector of metrics from a single file
#  @param fn The path to the .csv file
#  @return df The loaded DataFrame with a column header of the subject's ID
def load_vector_file(fn):
    subjId = fn.split("/")[-3]
    df = pd.read_csv(fn)
    df = df.rename(columns={"0":subjId})
    return df

## Load a matrix of metrics from a single file
#  @param fn The path to the .csv file
#  @return df The flattened matrix as a DataFrame with a column header of the subject's ID
def load_matrix_file(fn):
    subjId = fn.split("/")[-3]
    mat = np.loadtxt(open(fn, "r"),  delimiter=",")
    df = pd.DataFrame(data=mat.flatten().T, columns=[subjId])
    return df


## Load a set of metadata (demographics) about a group of subjects
#  @param fn The path to the .csv file
#  @param group String representing group (population)
#  @return df The demographic data as a DataFrame
def load_metadata_file(fn, group):
    df = pd.read_csv(fn)

    # Ensure consistency across loaded metadata 
    # Gender
    if 'Fem=0 Male=1' in list(df):
        df = df.rename(columns={'Fem=0 Male=1': 'Sex'})

    # Age Group
    df['Age_Group'] = [group for i in range(df.shape[0])]

    # Joint Groups - might remove
    df['Sex/Age'] = df['Sex'] + ' ' + df['Age At Scan'].astype(str)

    # Drop an unnecessary index column
    if 'Unnamed: 0' in list(df):
        df = df.drop(columns=['Unnamed: 0'])

    return df

## Created a cleaned DataFrame from a list of disjoint DataFrames
#  @param dataList A list of DataFrames to combine and clean
#  @return df The cleaned DataFrame
def combine_and_clean(dataList):
    # Convert the list of DataFrames into a single DataFrame
    df = pd.concat(dataList, axis=1)

    # Deal with NAN elements
    df = df.fillna(pd.DataFrame.max(df))

    # Remove duplicate columns
    df = df.T.drop_duplicates().T

    return df

##
# 
def union_of_dfs(df1, df2):
    pass

## Load the same metrics file for a group of subjects and store in DataFrame
#  @param subjects List of subjects
#  @param metricFn String specifying metrics file
#  @param metricType String specifying vector or matrix
def load_population_metrics(subjects, metricFn, isVector=True):

    metrics = []

    # Iterate through the subjects
    for subjectPath in subjects:
        fn = os.path.join(subjectPath, metricFn)

        # Load the file
        if isVector:
            tmpDf = load_vector_file(fn)
        else:
            tmpDf = load_matrix_file(fn)

        # Add the loaded DataFrame to the list of metrics
        metrics.append(tmpDf)

    # Clean up the loaded metrics and convert to a DataFrame
    df = combine_and_clean(metrics)

    return df

