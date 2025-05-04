import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Ustawienie większej figury dla wszystkich wykresów
plt.figure(figsize=(20, 20))
plt.suptitle('Analiza klasteryzacji danych kołowych', fontsize=18)

# Wczytanie danych
data = pd.read_csv('../../data/lab6/circle.csv')
X = data.values

# Obliczenie liczby wykresów
# 1 dla danych wejściowych + 1 dla K-Means + 4 dla Hierarchicznej + 1 dla DBSCAN 
# + 4 dla GMM + 2 dla Spectral + 1 dla DBSCAN alt + 3 dla wykresów metryk
total_plots = 16
rows = 4
cols = 4

# Wizualizacja danych wejściowych
plt.subplot(rows, cols, 1)
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.7)
plt.title('Dane wejściowe')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)

current_plot = 2  # Zaczynamy od drugiego plotu

# Funkcja do oceny jakości klasteryzacji
def evaluate_clustering(X, labels, model_name):
    if len(np.unique(labels)) <= 1:
        print(f"{model_name} utworzył tylko jeden klaster - nie można obliczyć metryk")
        return None

    metrics = {}
    metrics['silhouette'] = silhouette_score(X, labels)
    metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
    
    print(f"\nMetryki dla {model_name}:")
    print(f"Silhouette Score: {metrics['silhouette']:.4f} (wyższy = lepszy, max 1)")
    print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz']:.4f} (wyższy = lepszy)")
    print(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.4f} (niższy = lepszy)")
    
    return metrics

# Funkcja do wizualizacji wyników klasteryzacji
def visualize_clustering(X, labels, model_name, metrics=None, plot_idx=None):
    global current_plot
    
    if plot_idx is None:
        plot_idx = current_plot
        current_plot += 1
    
    plt.subplot(rows, cols, plot_idx)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    
    title = f'Klasteryzacja - {model_name}'
    if metrics:
        title += f'\nS: {metrics["silhouette"]:.2f}, CH: {metrics["calinski_harabasz"]:.0f}, DB: {metrics["davies_bouldin"]:.2f}'
    
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.colorbar(label='Klaster')

# Funkcja pomocnicza do znalezienia optymalnego parametru eps dla DBSCAN
def find_optimal_eps(X, k=5, plot_idx=None):
    global current_plot
    if plot_idx is None:
        plot_idx = current_plot
        current_plot += 1
    
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    distances = np.sort(distances[:, k-1])
    
    plt.subplot(rows, cols, plot_idx)
    plt.plot(distances)
    plt.title(f'Wykres odległości do {k}-tego sąsiada')
    plt.xlabel('Punkty posortowane wg odległości')
    plt.ylabel(f'Odległość do {k}-tego sąsiada')
    plt.grid(True)
    
    # Znajdź "łokieć" na wykresie (punkt gdzie nachylenie się zmienia)
    # Uproszczona metoda: znajdź punkt gdzie druga pochodna osiąga maksimum
    diffs = np.diff(distances)
    diffs2 = np.diff(diffs)
    elbow_idx = np.argmax(diffs2) + 2  # +2 bo mamy dwie różnice
    optimal_eps = distances[elbow_idx]
    
    print(f"Optymalny parametr eps dla DBSCAN: {optimal_eps:.4f}")
    return optimal_eps

# Funkcja do oceny optymalnej liczby klastrów dla metod parametrycznych
def find_optimal_k(X, max_k=10):
    global current_plot
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        silhouette_scores.append(silhouette_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))
    
    # Wizualizacja wyników - trzy metryki na jednym wykresie
    plot_idx = current_plot
    current_plot += 1
    
    plt.subplot(rows, cols, plot_idx)
    plt.plot(k_values, silhouette_scores, 'o-', label='Silhouette')
    plt.plot(k_values, [s/max(ch_scores) for s in ch_scores], 'o-', label='CH (norm)')
    plt.plot(k_values, [1-s/max(db_scores) for s in db_scores], 'o-', label='1-DB (norm)')
    plt.title('Metryki jakości klastrów')
    plt.xlabel('Liczba klastrów')
    plt.legend()
    plt.grid(True)
    
    # Znajdź optymalne wartości na podstawie metryk
    optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
    optimal_k_ch = k_values[np.argmax(ch_scores)]
    optimal_k_db = k_values[np.argmin(db_scores)]
    
    print(f"Optymalna liczba klastrów według Silhouette Score: {optimal_k_silhouette}")
    print(f"Optymalna liczba klastrów według Calinski-Harabasz Index: {optimal_k_ch}")
    print(f"Optymalna liczba klastrów według Davies-Bouldin Index: {optimal_k_db}")
    
    return optimal_k_silhouette, optimal_k_ch, optimal_k_db

# Analiza optymalnej liczby klastrów
k_silhouette, k_ch, k_db = find_optimal_k(X)

# Znalezienie optymalnego eps dla DBSCAN
optimal_eps = find_optimal_eps(X)

# 1. K-Means (z optymalną liczbą klastrów)
print("\n--- K-Means ---")
n_clusters = k_silhouette  # Używamy liczby klastrów optymalnej według Silhouette Score
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)
kmeans_metrics = evaluate_clustering(X, kmeans_labels, f"K-Means (k={n_clusters})")
visualize_clustering(X, kmeans_labels, f"K-Means (k={n_clusters})", kmeans_metrics)

# 2. Hierarchiczna aglomeracja
print("\n--- Hierarchiczna aglomeracja ---")
for linkage in ['ward', 'complete', 'average', 'single']:
    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    agglo_labels = agglo.fit_predict(X)
    agglo_metrics = evaluate_clustering(X, agglo_labels, f"Hierarchiczna ({linkage})")
    visualize_clustering(X, agglo_labels, f"Hierarchiczna ({linkage})", agglo_metrics)

# 3. DBSCAN
print("\n--- DBSCAN ---")
dbscan = DBSCAN(eps=optimal_eps, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"DBSCAN znalazł {n_clusters_dbscan} klastrów i {np.sum(dbscan_labels == -1)} punktów szumu")
if len(set(dbscan_labels)) > 1:  # Jeśli znaleziono więcej niż jeden klaster
    dbscan_metrics = evaluate_clustering(X, dbscan_labels, "DBSCAN")
    visualize_clustering(X, dbscan_labels, "DBSCAN", dbscan_metrics)
else:
    visualize_clustering(X, dbscan_labels, "DBSCAN")

# 4. Gaussian Mixture Model
print("\n--- Gaussian Mixture Model ---")
for covariance_type in ['full', 'tied', 'diag', 'spherical']:
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, random_state=42)
    gmm_labels = gmm.fit_predict(X)
    gmm_metrics = evaluate_clustering(X, gmm_labels, f"GMM ({covariance_type})")
    visualize_clustering(X, gmm_labels, f"GMM ({covariance_type})", gmm_metrics)

# 5. Spectral Clustering
print("\n--- Spectral Clustering ---")
for affinity in ['rbf', 'nearest_neighbors']:
    spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=42)
    spectral_labels = spectral.fit_predict(X)
    spectral_metrics = evaluate_clustering(X, spectral_labels, f"Spectral ({affinity})")
    visualize_clustering(X, spectral_labels, f"Spectral ({affinity})", spectral_metrics)

# W przypadku dbscan możemy również sprawdzić inną kombinację parametrów
print("\n--- DBSCAN z alternatywnymi parametrami ---")
dbscan_alt = DBSCAN(eps=optimal_eps*0.8, min_samples=4)  # Mniejszy eps, mniej punktów min_samples
dbscan_alt_labels = dbscan_alt.fit_predict(X)
n_clusters_dbscan_alt = len(set(dbscan_alt_labels)) - (1 if -1 in dbscan_alt_labels else 0)
print(f"DBSCAN (alt) znalazł {n_clusters_dbscan_alt} klastrów i {np.sum(dbscan_alt_labels == -1)} punktów szumu")
if len(set(dbscan_alt_labels)) > 1:
    dbscan_alt_metrics = evaluate_clustering(X, dbscan_alt_labels, "DBSCAN (alt)")
    visualize_clustering(X, dbscan_alt_labels, "DBSCAN (alt)", dbscan_alt_metrics)
else:
    visualize_clustering(X, dbscan_alt_labels, "DBSCAN (alt)")

# Podsumowanie wyników
print("\n--- Podsumowanie wyników ---")
print("Najlepsze wyniki dla tego zbioru danych uzyskano przy użyciu algorytmów:")
print("1. DBSCAN - który dobrze radzi sobie z wykrywaniem klastrów o dowolnych kształtach")
print("2. Spectral Clustering - który wykorzystuje podobieństwo między punktami, a nie tylko odległości")
print("3. GMM - który modeluje dane jako mieszaninę rozkładów normalnych")

# Dostosowanie układu
plt.tight_layout(rect=[0, 0, 1, 0.97])  # Drobna korekta dla tytułu
plt.show()
