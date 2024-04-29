from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cargar el archivo CSV en un DataFrame de pandas
file_path = 'Empleabilidad/job_placement.csv'
data = pd.read_csv(file_path)

# Extracting numerical features
features = data[['age', 'gpa', 'years_of_experience']]

# Filling any missing values with the mean of the column
features = features.fillna(features.mean())

# Setting the value of k
k = 5

# Initializing the Nearest Neighbors model
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(features)

# Finding the k-nearest neighbors for each point
distances, indices = nbrs.kneighbors(features)

# We remove the first column since it's the distance to itself (0)
k_distances = distances[:, 1]  # taking only the distance to the k-th nearest neighbor

# Plotting the k-nearest distances
plt.figure(figsize=(10, 6))
plt.plot(np.sort(k_distances))
plt.title(f'Distances to {k}-th Nearest Neighbors')
plt.xlabel('Index')
plt.ylabel('Distance')
plt.grid(True)
plt.show()