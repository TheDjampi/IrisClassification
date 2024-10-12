# Import des bibliothèques nécessaires
import pandas as pd
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Charger le jeu de données Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Étape 1 : Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Étape 2 : Calcul des distances et construction du dendrogramme
Z = linkage(X_scaled, method='ward')

# Étape 3 : Visualisation du dendrogramme
plt.figure(figsize=(10, 7))
plt.title("Dendrogramme de classification hiérarchique (méthode de Ward)")
dendrogram(Z, labels=iris.target_names[y])
plt.xlabel("Espèces")
plt.ylabel("Distance")
plt.show()

# Étape 4 : Découper le dendrogramme pour obtenir des clusters
# Ici, on choisit arbitrairement 3 clusters (car on sait que les données Iris ont 3 classes)
clusters = fcluster(Z, 3, criterion='maxclust')

# Étape 5 : Évaluer la qualité du clustering
# Silhouette score
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score : {silhouette_avg:.2f}")

# Comparaison avec les vraies étiquettes
adjusted_rand = adjusted_rand_score(y, clusters)
print(f"Adjusted Rand Index : {adjusted_rand:.2f}")
