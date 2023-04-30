import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from IPython.display import display

df = pd.read_csv("exoplanets_dataset.csv", skiprows=(range(46)))

features = df.loc[:, ["pl_rade", "pl_bmasse"]]

# Usuwanie outlierów
cutoff = 4131.4  # powyżej 4131.4 mas Ziemi -> brązowe karły
features = features[features["pl_bmasse"] <= cutoff]

# Skalowanie
scaler = StandardScaler()  # z = (x - u) / s
scaled_features = scaler.fit_transform(features)

kmeans_kwargs = {
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300}

silhouette_coefficients = []
sse = []

cluster_nos = [2, 3, 4, 6]  # n dla których chcemy utworzyć wykresy punktowe
cluster_labels = []

# Obliczanie najlepszej ilości klastrów
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    if k in cluster_nos:
        cluster_labels.append(kmeans.predict(scaled_features))
    sse.append(kmeans.inertia_)
    if k > 1:
        score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_coefficients.append(score)

figsize = [10.24, 7.68]

# Wyniki dla metody Silhoutte Coefficient
plt.figure(figsize=figsize)
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.title("Silhoutte Coefficient")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.grid(alpha=0.3)
plt.show()

# Wyniki dla metody Elbow Method
plt.figure(figsize=figsize)
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.grid(alpha=0.3)
plt.show()

# Dla każdego k w cluster_nos narysuj wykres
for k in cluster_nos:
    k_labels = cluster_labels[cluster_nos.index(k)]
    plt.figure(figsize=figsize)
    sc = plt.scatter(features["pl_rade"], features["pl_bmasse"], c=k_labels, cmap="plasma", s=20, alpha=0.4)
    plt.legend(*sc.legend_elements(), title='Klastry', loc="lower right")
    plt.title(f"Analiza skupień dla n={k}")
    plt.xlabel("Promień [Promień Ziemi]")
    plt.ylabel("Masa [Masa Ziemi]")
    plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.show()

    # Annotacje planet
    # features2 = df.loc[:, ["pl_name", "pl_rade", "pl_bmasse"]]
    # for x in range(len(features2)):
    #     plt.annotate(text=features2.loc[x].at["pl_name"],
    #                  xy=(features2.loc[x].at["pl_rade"], features2.loc[x].at["pl_bmasse"]),
    #                  fontsize='xx-small',
    #                  alpha=0.3)

    # Printuj liczebności klastrów
    unique_labels, label_counts = np.unique(k_labels, return_counts=True)
    print(f"\nLiczności klastrów dla n={k}:")
    for label, count in zip(unique_labels, label_counts):
        print("Cluster {}: {} objects".format(label, count))

    # Histogramy liczności klastrów <- do zrobienia
    # plt.hist(k_labels, bins=len(unique_labels))
    # plt.xticks(unique_labels)
    # plt.xlabel("Cluster label")
    # plt.ylabel("Number of objects")
    # plt.show()