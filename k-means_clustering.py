import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from IPython.display import display

df = pd.read_csv("exoplanets_dataset.csv", skiprows=(range(46)))

features = df.loc[:, ["pl_rade", "pl_bmasse"]]

# Usuwanie outlierów
mass_cutoff = 4131.4  # powyżej 4131.4 mas Ziemi -> brązowe karły
radius_cutoff = 24.66 # 24.66 -> promień obiektu CT Cha b

features = features.loc[(features['pl_bmasse'] < mass_cutoff) & (features['pl_rade'] < radius_cutoff)]
print(len(features))

# Skalowanie
scaler = Scaler()  # z = (x - u) / s
scaled_features = scaler.fit_transform(features)

kmeans_kwargs = {
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300}

silhouette_scores = []
sse_scores = []
calinski_scores = []
davies_scores = []

cluster_nos = [3, 4, 5, 6, 7, 10]  # n dla których chcemy utworzyć wykresy punktowe
cluster_labels = []

# Obliczanie najlepszej ilości klastrów
for k in range(1, 13):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)

    if k in cluster_nos:
        cluster_labels.append(kmeans.predict(scaled_features))

    sse_scores.append(kmeans.inertia_)
    if k > 1:
        sil_score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_scores.append(sil_score)
        ch_score = calinski_harabasz_score(scaled_features, kmeans.labels_)
        calinski_scores.append(ch_score)
        db_score = davies_bouldin_score(scaled_features, kmeans.labels_)
        davies_scores.append(db_score)

figsize = [10.24, 7.68]

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)

axs[0, 0].plot(range(1, 13), sse_scores)
axs[0, 0].set_title("Elbow Method")
axs[0, 0].set_xlabel("Liczba klastrów")
axs[0, 0].set_ylabel("SSE")
axs[0, 0].set_xticks(range(1, 13))
axs[0, 0].grid(alpha=0.3)

axs[0, 1].plot(range(2, 13), silhouette_scores)
axs[0, 1].set_title("Silhouette Coefficient")
axs[0, 1].set_xlabel("Liczba klastrów")
axs[0, 1].set_ylabel("Silhouette Score")
axs[0, 1].set_xticks(range(2, 13))
axs[0, 1].grid(alpha=0.3)

axs[1, 0].plot(range(2, 13), calinski_scores)
axs[1, 0].set_title("Indeks Calińskiego i Harabasza")
axs[1, 0].set_xlabel("Liczba klastrów")
axs[1, 0].set_ylabel("Calinski Harabasz Score")
axs[1, 0].set_xticks(range(2, 13))
axs[1, 0].grid(alpha=0.3)

axs[1, 1].plot(range(2, 13), davies_scores)
axs[1, 1].set_title("Indeks Daviesa i Bouldina")
axs[1, 1].set_xlabel("Liczba klastrów")
axs[1, 1].set_ylabel("Davies Bouldin Score")
axs[1, 1].set_xticks(range(2, 13))
axs[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Dla każdego k w cluster_nos narysuj wykres
for k in cluster_nos:
    k_labels = cluster_labels[cluster_nos.index(k)]
    plt.figure(figsize=figsize)
    sc = plt.scatter(features["pl_rade"], features["pl_bmasse"], c=k_labels, cmap="plasma", s=20, alpha=0.4)
    plt.legend(*sc.legend_elements(), title='Klastry', loc="lower right")
    plt.title(f"Analiza skupień k-średnich dla n={k}")
    plt.xlabel("Promień [Promień Ziemi]")
    plt.ylabel("Masa [Masa Ziemi]")
    plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    scatter_colors = sc.to_rgba(k_labels)

    # Adnotacje planet - uwaga mogą lagować
    # features_annotated = df.loc[:, ["pl_name", "pl_rade", "pl_bmasse"]]
    # for x in range(len(features_annotated)):
    #     plt.annotate(text=features_annotated.loc[x].at["pl_name"],
    #                  xy=(features_annotated.loc[x].at["pl_rade"], features_annotated.loc[x].at["pl_bmasse"]),
    #                  fontsize='xx-small',
    #                  alpha=0.3)

    plt.show()

    # Wykresy słupkowe liczności klastrów
    unique_labels, label_counts = np.unique(k_labels, return_counts=True)
    colors = sc.to_rgba(k_labels)
    unique_colors = [colors[np.where(k_labels == label)[0][0]] for label in unique_labels]

    fig, ax = plt.subplots()
    bars = ax.bar(unique_labels, label_counts, color=unique_colors)
    ax.set_xticks(unique_labels)
    ax.set_xlabel("Etykieta klastra")
    ax.set_ylabel("Liczba obiektów")
    ax.bar_label(bars)
    fig.set_tight_layout(True)
    plt.show()