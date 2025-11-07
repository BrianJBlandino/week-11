import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
from time import time
import matplotlib.pyplot as plt

def kmeans(X, k):
    """Creating a function that performs k-means clustering
    on a numerical NumPy array X that returns a tuple where
    centroids is a 2D array of shape containing the cluster
    centroids, and labels is a 1D array of shape containing
    the index of the assigned cluster for each row of X."""
    
    # Initialize and fit the Scikit-Learn KMeans model
    model = KMeans(n_clusters=k, n_init='auto', random_state=42)
    model.fit(X)
    
    # Extracting the centroids and labels as NumPy arrays
    centroids = model.cluster_centers_
    labels = model.labels_
    
    return centroids, labels

def kmeans_diamonds(n, k):
    """Creating a function that runs a prior kmeans function
    to create k clusters on the first n rows of the (numeric)
    diamonds dataset."""
    
    # Load the dataset
    diamonds = sns.load_dataset("diamonds")
    
    # Keep only numerical columns
    numeric_diamonds = diamonds.select_dtypes(include='number')
    
    # Restricting to the first n rows
    X = numeric_diamonds.head(n).to_numpy()

    # Using the kmeans() function defined in Exercise 1
    centroids, labels = kmeans(X, k)

    # Returning results
    return centroids, labels

def kmeans_timer(n, k, n_iter=5):
    """Creating a function that runs the function
    kmeans_diamonds(n, k) exactly n_iter times, and saves the
    runtime for reach run and returns the average time across
    the n runs, where 'time' is in seconds."""
    
    # Creating an empty list to store the runtimes
    times = []
    
    for _ in range(n_iter):
        start = time()
        centroids, labels = kmeans_diamonds(n, k)
        end = time()
        elapsed = end - start
        times.append(elapsed)
    
    avg_time = np.mean(times)

    # Returning both the list of individual runtimes and the average
    return times, avg_time