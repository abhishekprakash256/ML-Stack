import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/NYCTaxiFares.csv")
#
#print(df['fare_amount'].describe())

def haversine_distance(df, lat1, long1, lat2, long2):
    """
    Calculates the haversine distance between 2 sets of GPS coordinates in df
    """
    r = 6371  # average radius of Earth in kilometers

    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])

    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (r * c) # in kilometers

    return d

#create a new column and make the data for it, by adding the feature engineering
df['dist_km'] = haversine_distance(df,'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
#print(df.head())
#
#conversion to datetime object
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
#print(df.info())
print(df['pickup_datetime'][1])
