import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# wczytanie danych z pliku csv
data = pd.read_excel('C:/Users/acer/Desktop/PSC.xlsx')

# wybór kolumn do klastrowania
X = data[['pl_rade', 'pl_bmasse', 'sy_dist']].values

# usunięcie wartości NaN
X = np.nan_to_num(X)

# standaryzacja danych
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# klastrowanie hierarchiczne z użyciem metody Warda
Z = linkage(X, method='ward')

# wyświetlenie dendrogramu
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dendrogram(Z)

plt.show()

# wyświetlenie klastrów
k = 2  # liczba klastrów
clusters = fcluster(Z, k, criterion='maxclust')

# wyświetlenie wyników klastrowania
results = pd.DataFrame({'pl_name': data['pl_name'], 'cluster': clusters})
print(results)
