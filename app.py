import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("k-Means Clustering Demo mit Streamlit")

# Slider für die Anzahl der Datenpunkte
num_points = st.slider("Anzahl der Datenpunkte", 10, 200, 50) # Erhöhtes Maximum für mehr Variation

# Slider für den Bereich der Zufallszahlen
min_val = st.number_input("Minimaler Wert für die Datenpunkte", value=0)
max_val = st.number_input("Maximaler Wert für die Datenpunkte", value=100)

# Datengenerierung (Kein Seed mehr!)
data = np.random.randint(min_val, max_val + 1, num_points) #+1 da randint exklusiv ist
data = data.reshape(-1, 1)

if len(np.unique(data))<2:
    st.warning("Es wurden nur identische Datenpunkte generiert. Bitte ändere den Wertebereich oder erhöhe die Anzahl der Datenpunkte.")
else:

    # Elbow-Methode (angepasst für den Fall weniger Datenpunkte)
    inertia = []
    max_k = min(10, len(np.unique(data)))  # Maximal 10 Cluster oder Anzahl unterschiedlicher Datenpunkte
    K = range(1, max_k + 1)
    if len(K)==1:
        st.warning("Es gibt nur einen eindeutigen Datenpunkt, Clustering nicht möglich.")
    else:
        for k in K:
            kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10)
            kmeans.fit(data)
            inertia.append(kmeans.inertia_)


        # Plot des Elbow-Diagramms in Streamlit
        st.subheader("Elbow-Methode zur Bestimmung von k")
        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(K, inertia, 'bx-')
        ax_elbow.set_xlabel('Anzahl der Cluster (k)')
        ax_elbow.set_ylabel('Inertia')
        ax_elbow.set_xticks(K) #Stellt sicher, dass alle k-Werte auf der X-Achse angezeigt werden
        st.pyplot(fig_elbow)

        # Eingabefeld für den optimalen Wert von k
        optimal_k = st.number_input("Optimaler Wert für k (aus dem Elbow-Diagramm auswählen)", min_value=1, max_value=len(np.unique(data)), value=min(2, len(np.unique(data))), step=1)

        if optimal_k > len(np.unique(data)):
            st.warning("Der Wert für k ist größer als die Anzahl eindeutiger Datenpunkte. Clustering kann zu unerwarteten Ergebnissen führen.")
        else:
            # k-Means-Anwendung
            kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10)
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
