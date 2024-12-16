import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Erzeugung von 10 zufälligen Datenpunkten (1-dimensional)
data = np.random.randint(0, 100, 10)
data = data.reshape(-1, 1)

# 2. Bestimmung der optimalen Anzahl von Clustern (k) mit der Elbow-Methode
inertia = []
K = range(1, min(len(data),10))

for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10) # Kein random_state mehr!
    kmeans.fit(data)
    inertia.append(kmeans.inertia_)

# Plot des Elbow-Diagramms
plt.plot(K, inertia, 'bx-')
plt.xlabel('Anzahl der Cluster (k)')
plt.ylabel('Inertia')
plt.title('Elbow-Methode zur Bestimmung von k')
plt.show()

# 3. Anwendung von k-Means mit der optimalen Anzahl von Clustern
optimal_k = 2  # Diesen Wert ggf. aus dem Elbow-Diagramm anpassen

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10) # Kein random_state mehr!
kmeans.fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 4. Visualisierung der Ergebnisse
plt.scatter(data, np.zeros_like(data), c=labels, s=100)
plt.scatter(centroids, np.zeros_like(centroids), marker='x', s=200, linewidths=3, color='r')
plt.xlabel('Datenwerte')
plt.title(f'k-Means Clustering mit k={optimal_k}')
plt.show()

# Ausgabe der Clusterzugehörigkeiten und Zentroiden
print("Datenpunkte:", data.flatten())
print("Clusterzugehörigkeiten:", labels)
print("Zentroiden:", centroids.flatten())
