#!/usr/bin/env python

# Core
import re, csv, requests, math
import pandas as pd
from pandas.compat import StringIO
import scipy as sp
import numpy as np
# Plotting
import missingno as msno
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
sns.set(style="ticks")
# Machine learning
import sklearn as skl
from sklearn import svm, datasets
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score


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
def modes(df, key_cols, value_col, count_col):
    '''                                                                                                                                                                                                                                                                                                                                                              
    Pandas does not provide a `mode` aggregation function                                                                                                                                                                                                                                                                                                            
    for its `GroupBy` objects. This function is meant to fill                                                                                                                                                                                                                                                                                                        
    that gap, though the semantics are not exactly the same.                                                                                                                                                                                                                                                                                                         

    The input is a DataFrame with the columns `key_cols`                                                                                                                                                                                                                                                                                                             
    that you would like to group on, and the column                                                                                                                                                                                                                                                                                                                  
    `value_col` for which you would like to obtain the modes.                                                                                                                                                                                                                                                                                                        

    The output is a DataFrame with a record per group that has at least                                                                                                                                                                                                                                                                                              
    one mode (null values are not counted). The `key_cols` are included as                                                                                                                                                                                                                                                                                           
    columns, `value_col` contains lists indicating the modes for each group,                                                                                                                                                                                                                                                                                         
    and `count_col` indicates how many times each mode appeared in its group. 
    
    This was modified from StackExchange user abw333.
    '''
    return df.groupby(key_cols + [value_col]).size().to_frame(count_col).reset_index().groupby(key_cols + [count_col])[value_col].unique().reset_index()

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
    
    temp.dropna(subset=[rank]).reset_index(drop=True)


