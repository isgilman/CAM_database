#!/usr/bin/env python

# Core
import re, csv, requests, math
import pandas as pd
from pandas.compat import StringIO
import scipy as sp
import numpy as np
# Plotting
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import missingno as msno
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def pull_data(dataset='cam_db'):
    
    if dataset=='cam_db':
        url = 'https://raw.githubusercontent.com/isgilman/CAM_database/master/CAM_database.csv'  
    elif dataset=='the_list':
        url = 'https://raw.githubusercontent.com/isgilman/CAM_database/master/Genera_Photosynthesis.csv'
    
    gitdata = requests.get(url).text
    df = pd.read_csv(StringIO(gitdata))

    for col in df.columns:
    	if "Unnamed" in col:
        	df.drop(labels=[col], axis=1, inplace=True)

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def db_info(cam_db, plt_missingdata=False, **kwargs):
    '''Takes a pandas DataFrame made from the CAM database and returns some metrics on the 
    number of observations at different taxonomic ranks.
    
    Parameters
    ----------
    cam_db : pandas DataFrame containing CAM database from isgilman GitHub'''
    
    
    try:
        fams = len(cam_db.dropna(subset=[['Family']]).Family.unique())
    except KeyError:
        fams = None
    try:
        subfams = len(cam_db.dropna(subset=[['Subfamily']]).Subfamily.unique())
    except KeyError:
        subfams = None
    try:
        tribes = len(cam_db.dropna(subset=[['Tribe']]).Tribe.unique())
    except KeyError:
        tribes = None
    try:
        subtribes = len(cam_db.dropna(subset=[['Subtribe']]).Subtribe.unique())
    except KeyError:
        subtribes = None
    try:
        genera = len(cam_db.dropna(subset=[['Genus']]).Genus.unique())
    except KeyError:
        genera = None
    try:
        species = len(cam_db[cam_db['Species'] != 'sp.'].long_name.unique())
    except KeyError:
        species = len(cam_db.dropna(subset=[['long_name']]).long_name.unique())
    try:
        sources = len(cam_db.dropna(subset=[['Source']]).Source.unique())
    except KeyError:
        sources = None

    print "The CAM database consists of:\n\t{} observations,\
                                        \n\t{} families,\
                                        \n\t{} subfamiles,\
                                        \n\t{} tribes,\
                                        \n\t{} subtribes,\
                                        \n\t{} genera, and\
                                        \n\t{} species from {} publications.\
            \n{:.3} of the data matrix is missing and\
            \n{} observations have known photosynthetic pathways.".format(
        len(cam_db),
        fams,
        subfams,
        tribes,
        subtribes,
        genera,
        species,
        sources,
        float(cam_db.melt().isnull().sum()[1])/(cam_db.shape[0]*cam_db.shape[1]),
        len(cam_db.dropna(subset=[['Pathway']])))
    
    if plt_missingdata:
        msno.matrix(cam_db, **kwargs)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_svc_decision_function(model, ax=None, plot_support=True):
    '''Plot the decision function for a 2D SVC. From the Python Data 
    Science Handbook by Jake VanderPlas'''
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def catmode(df, key_col, value_col, count_col='Count', tiebreak='random'):
    '''                                                                                                                                                                                                                                                                                                                                                              
    Pandas does not provide a `mode` aggregation function                                                                                                                                                                                                                                                                                                            
    for its `GroupBy` objects. This function is meant to fill                                                                                                                                                                                                                                                                                                        
    that gap, though the semantics are not exactly the same.                                                                                                                                                                                                                                                                                                         

    The input is a DataFrame with the columns `key_cols`                                                                                                                                                                                                                                                                                                             
    that you would like to group on, and the column                                                                                                                                                                                                                                                                                                                  
    `value_col` for which you would like to obtain the mode.                                                                                                                                                                                                                                                                                                         

    The output is a DataFrame with a record per group that has at least one mode                                                                                                                                                                                                                                                                                     
    (null values are not counted). The `key_cols` are included as columns, `value_col`                                                                                                                                                                                                                                                                               
    contains a mode (ties are broken arbitrarily and deterministically) for each                                                                                                                                                                                                                                                                                     
    group, and `count_col` indicates how many times each mode appeared in its group.   

    This fucntion was written by StackExchange user abw333 and altered slightly.                                                                                                                                                                                                                                                                           
    '''
    temp = df.groupby([key_col, value_col]).size().to_frame(count_col).reset_index().sort_values(count_col, ascending=False)

    ranks = []
    modes = []
    modeframe = pd.DataFrame()
    for rank in temp[key_col].unique():
        subset = temp.loc[temp[key_col]==rank]
        maxmode = max(subset['Count'])
        withmax = subset.loc[subset['Count']==maxmode]
        if len(withmax)>1:
            if tiebreak=='random':
                mode = np.random.choice(withmax[value_col], size=1)[0]
            elif tiebreak=='first':
                mode = withmax[value_col].tolist()[0]
            elif tiebreak=='last':
                mode = withmax[value_col].tolist()[-1]
            elif tiebreak=='none':
                mode = np.nan
        else:
            mode = withmax[value_col].tolist()[0]
        ranks.append(rank)
        modes.append(mode)

    modeframe[key_col] = ranks
    modeframe[value_col] = modes
    return modeframe

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def check_nan(thing):
    try: 
        return math.isnan(thing)
        
    except:
        TypeError
        return False

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def drop_ambig(DF, rank='Species', ambigs=['species', 'sp', 'spp', 'spec', 'sp.', 'spp.', 'spec.', 'hybrid']):
    
    temp = DF[~DF[rank].str.lower().isin([x.lower() for x in ambigs])]
    
    return temp.dropna(subset=[rank]).reset_index(drop=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def data_fill_modes(df, group_by, to_fill, tiebreak='random', debug=False):
    
    # Create a copy to prevent bias after randomly choosing among responses
    copy = df.copy()
    fill_dict={}
    # Loop over categories
    for cat in to_fill:
        filled_holes=0
        # Create temporary dataframe with category modes
        modeframe = catmode(df, key_col=group_by, value_col=cat)
        for index, row in copy.iterrows():
            taxon = copy.loc[index, group_by]
            # Skip unidentified taxa
            if taxon.lower() in ['species', 'sp', 'spp', 'spec', 'sp.', 'spp.', 'spec.']:
                if debug: print("UH OH. UNIDENTIFIED TAXON: {}".format(taxon))
                continue
            # Only fill if current value is missing
            current = copy.loc[index, cat]
            if debug: print("Current taxon: {}, pathway: {}".format(taxon, current))
            if check_nan(current) and taxon in modeframe[group_by].tolist():
                if debug: print("\tFOUND A HOLE IN {}".format(taxon))
                rankmode = modeframe.loc[modeframe[group_by]==taxon][cat].tolist()[0]
                if debug: print("\tFound a mode for {}: {}".format(taxon, rankmode))
            else:
                rankmode = np.NaN
                if debug: print("\t\tCANT FILL HOLE for {}: {}".format(taxon, rankmode))
            copy.loc[index, cat] = rankmode
    if debug: print fill_dict
    return copy

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def fill_numeric(df, group_by, values, how='mean'):
    
    if how=='mean':
        temp = df.groupby(group_by).transform(lambda x: x.fillna(x.mean()))[values]
        orig_cols = list(set(df.columns)-set(temp.columns))
    return pd.concat([df[orig_cols], temp], axis=1).reset_index(drop=True)
