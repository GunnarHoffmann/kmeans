import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("k-Means Clustering Demo mit Streamlit")

# Slider für die Anzahl der Datenpunkte
num_points = st.slider("Anzahl der Datenpunkte", 10, 100, 10)

# Slider für den Seed für die Zufallszahlen
seed = st.slider("Seed für Zufallszahlen (für Reproduzierbarkeit)", 0, 100, 0)
np.random.seed(seed)

# Datengenerierung
data = np.random.randint(0, 100, num_points)
data = data.reshape(-1, 1)

# Elbow-Methode
inertia = []
K = range(1, min(len(data),10))

for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=seed)
    kmeans.fit(data)
    inertia.append(kmeans.inertia_)

# Plot des Elbow-Diagramms in Streamlit
st.subheader("Elbow-Methode zur Bestimmung von k")
fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(K, inertia, 'bx-')
ax_elbow.set_xlabel('Anzahl der Cluster (k)')
ax_elbow.set_ylabel('Inertia')
st.pyplot(fig_elbow)

# Eingabefeld für den optimalen Wert von k
optimal_k = st.number_input("Optimaler Wert für k (aus dem Elbow-Diagramm auswählen)", min_value=1, max_value=len(data), value=2, step=1)

# k-Means-Anwendung
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=seed)
kmeans.fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualisierung der Ergebnisse in Streamlit
st.subheader(f"k-Means Clustering mit k={optimal_k}")
fig_kmeans, ax_kmeans = plt.subplots()
ax_kmeans.scatter(data, np.zeros_like(data), c=labels, s=100)
ax_kmeans.scatter(centroids, np.zeros_like(centroids), marker='x', s=200, linewidths=3, color='r')
ax_kmeans.set_xlabel('Datenwerte')
st.pyplot(fig_kmeans)

# Ausgabe der Ergebnisse in Streamlit
st.subheader("Ergebnisse")
st.write("Datenpunkte:", data.flatten())
st.write("Clusterzugehörigkeiten:", labels)
st.write("Zentroiden:", centroids.flatten())
