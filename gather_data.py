import pandas as pd
import numpy as np
import os

## Remove nonnumeric characters from the end of every line in a file
#  @param fn The path to the .csv file
def remove_trailing_commas(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()

    lines = [l.rstrip()[:-1] for l in lines]

    with open(fn, 'w') as f:
        for l in lines:
            f.write(l+"\n")

## Load a vector of metrics from a single file
#  @param fn The path to the .csv file
#  @return df The loaded DataFrame with a column header of the subject's ID
def load_vector_file(fn, length=None):
    subjId = fn.split("/")[-3]
    if length is not None:
        df = pd.read_csv(fn)[:length]
    else:
        df = pd.read_csv(fn)
    df = df.rename(columns={"0":subjId})
    return df

## Load a matrix of metrics from a single file
#  @param fn The path to the .csv file
#  @return df The flattened matrix as a DataFrame with a column header of the subject's ID
def load_matrix_file(fn, length=None):
    subjId = fn.split("/")[-3]
    if length is not None:
        try:
            mat = np.loadtxt(open(fn, "r"),  delimiter=",")[:length, :length]
        except ValueError:
            remove_trailing_commas(fn)
            mat = np.loadtxt(open(fn, "r"), delimiter=",")[:length, :length]
    else:
        mat = np.loadtxt(open(fn, "r"), delimiter=",")
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
    if 'Sex' in list(df) and 'Age At Scan' in list(df):
        df['Sex/Age'] = df['Sex'] + ' ' + df['Age At Scan'].astype(str)

    # Drop an unnecessary index column
    if 'Unnamed: 0' in list(df):
        df = df.drop(columns=['Unnamed: 0'])

    # Transpose the demographics dataframe and make the column headers subject IDs
    df = df.set_index('ID').T

    return df

## Create a cleaned DataFrame from a list of disjoint DataFrames
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

## Only keep the columns from each dataframe that are present in both dataframes
#  @param df Dataframe to filter
#  @param columns List of columns to keep
#  @return df Dataframe with only shared columns
def keep_shared_columns(df, columns):
    # Keep only the shared columns
    df = df[columns]
        
    return df

## Load the same metrics file for a group of subjects and store in DataFrame
#  @param subjects List of subjects
#  @param metricFn String specifying metrics file
#  @param length Integer specifying the size of the metrics to load 
#  @param metricType String specifying vector or matrix
def load_population_metrics(subjects, metricFn, length, isVector=True):

    metrics = []

    # Iterate through the subjects
    for subjectPath in subjects:
        fn = os.path.join(subjectPath, metricFn)

        # Load the file
        if isVector:
            tmpDf = load_vector_file(fn, length)
        else:
            tmpDf = load_matrix_file(fn, length)

        # Add the loaded DataFrame to the list of metrics
        metrics.append(tmpDf)

    # Clean up the loaded metrics and convert to a DataFrame
    df = combine_and_clean(metrics)

    return df

