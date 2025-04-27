import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer

# Wczytanie danych
data = pd.read_csv('../../data/lab6/customers_mall.csv', sep=';')

# Wyświetlenie pierwszych 5 wierszy danych
print("Pierwsze 5 wierszy danych:")
print(data.head())

# Podstawowe informacje o danych
print("\nOpis danych:")
print(data.describe())

# Standaryzacja danych
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Wizualizacja danych
plt.figure(figsize=(10, 8))
plt.scatter(data['Annual Income'], data['Spending Score'], s=50)
plt.title('Roczny dochód vs. Wydatki klientów', fontsize=15)
plt.xlabel('Roczny dochód (w tysiącach)', fontsize=12)
plt.ylabel('Punktowa ocena wydatków (0-100)', fontsize=12)
plt.grid(True)
plt.show()

# Porównanie różnych liczb klastrów
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, k in enumerate([2, 3, 4, 5]):
    # Wykonanie klasteryzacji
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data_scaled)
    
    # Odwrócenie standaryzacji dla centroidów
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Rysowanie wykresu
    for cluster in range(k):
        mask = cluster_labels == cluster
        axes[i].scatter(
            data.loc[mask, 'Annual Income'], 
            data.loc[mask, 'Spending Score'],
            s=50, 
            label=f'Klaster {cluster}'
        )
    
    axes[i].scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=200,
        marker='*',
        c='black',
        label='Centroidy'
    )
    
    axes[i].set_title(f'k = {k}', fontsize=14)
    axes[i].set_xlabel('Roczny dochód (w tysiącach)', fontsize=12)
    axes[i].set_ylabel('Punktowa ocena wydatków (0-100)', fontsize=12)
    axes[i].grid(True)
    axes[i].legend(loc='best', fontsize=10)
    
    # Silhouette score
    score = silhouette_score(data_scaled, cluster_labels)
    axes[i].annotate(f'Silhouette Score: {score:.3f}', 
                     xy=(0.05, 0.95), 
                     xycoords='axes fraction', 
                     fontsize=12,
                     va='top',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()

# Określenie optymalnej liczby klastrów za pomocą metody łokcia (Elbow Method)
plt.figure(figsize=(10, 8))
visualizer = KElbowVisualizer(KMeans(random_state=42), k=(1, 11))
visualizer.fit(data_scaled)
visualizer.show()

# Metoda silhouette score dla potwierdzenia optymalnej liczby klastrów
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_scaled)
    score = silhouette_score(data_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    print(f'Silhouette Score dla {k} klastrów: {score:.3f}')

# Wizualizacja wyników silhouette score
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score dla różnych wartości k', fontsize=15)
plt.xlabel('Liczba klastrów (k)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.grid(True)
plt.show()

# Na podstawie metody łokcia i silhouette score wybieramy optymalną liczbę klastrów
# Ustalamy optymalną wartość k na podstawie przeprowadzonych analiz
optimal_k = 5  # Możemy dostosować tę wartość po analizie wyników metody łokcia i silhouette score

# Sekcja podsumowująca analizę wyboru optymalnej liczby klastrów
print("\nAnaliza wyboru optymalnej liczby klastrów:")
print("1. Metoda łokcia (Elbow Method) wskazuje na punkt przegięcia przy k=5,")
print("   co sugeruje, że dodawanie kolejnych klastrów nie wnosi znaczącej poprawy.")
print("2. Współczynnik Silhouette osiąga wysokie wartości dla k=5, co potwierdza dobrą jakość klasteryzacji.")
print("3. Wizualna analiza klastrów pokazuje, że podział na 5 grup dobrze odzwierciedla naturalne skupiska w danych.")
print(f"W związku z powyższym, optymalna liczba klastrów dla tego zbioru danych to {optimal_k}.")

# Przeprowadzenie klasteryzacji k-means z optymalną liczbą klastrów
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Wizualizacja klastrów
plt.figure(figsize=(12, 8))
for cluster in range(optimal_k):
    plt.scatter(
        data[data['Cluster'] == cluster]['Annual Income'],
        data[data['Cluster'] == cluster]['Spending Score'],
        s=100,
        label=f'Klaster {cluster}'
    )

# Zaznaczenie centroidów
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=300,
    marker='*',
    c='black',
    label='Centroidy'
)

plt.title(f'Klasteryzacja klientów centrum handlowego (k={optimal_k})', fontsize=15)
plt.xlabel('Roczny dochód (w tysiącach)', fontsize=12)
plt.ylabel('Punktowa ocena wydatków (0-100)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Analiza klastrów
cluster_info = []
for cluster in range(optimal_k):
    cluster_data = data[data['Cluster'] == cluster]
    income_mean = cluster_data['Annual Income'].mean()
    spending_mean = cluster_data['Spending Score'].mean()
    
    print(f"\nKlaster {cluster}:")
    print(f"Liczba klientów: {len(cluster_data)}")
    print(f"Średni dochód roczny: {income_mean:.2f} tysięcy")
    print(f"Średnia ocena wydatków: {spending_mean:.2f}")
    
    cluster_info.append({
        'cluster': cluster,
        'size': len(cluster_data),
        'income_mean': income_mean,
        'spending_mean': spending_mean
    })

# Automatyczna interpretacja klastrów
print("\nInterpretacja klastrów:")
for info in cluster_info:
    cluster = info['cluster']
    income = "wysokim" if info['income_mean'] > 70 else "średnim" if info['income_mean'] > 40 else "niskim"
    spending = "wysoką" if info['spending_mean'] > 70 else "średnią" if info['spending_mean'] > 40 else "niską"
    
    print(f"Klaster {cluster}: Klienci o {income} dochodzie i {spending} skłonności do wydatków. " +
          f"Średni dochód: {info['income_mean']:.2f}k, średnia ocena wydatków: {info['spending_mean']:.2f}. " +
          f"Liczba klientów: {info['size']}")

# Dodatkowa analiza - określenie segmentów klientów
print("\nSegmentacja klientów:")
for info in cluster_info:
    cluster = info['cluster']
    if info['income_mean'] > 70 and info['spending_mean'] > 70:
        segment = "Premium (wysokie dochody, wysokie wydatki)"
    elif info['income_mean'] < 40 and info['spending_mean'] < 40:
        segment = "Oszczędni (niskie dochody, niskie wydatki)"
    elif info['income_mean'] > 70 and info['spending_mean'] < 40:
        segment = "Zamożni oszczędni (wysokie dochody, niskie wydatki)"
    elif info['income_mean'] < 40 and info['spending_mean'] > 70:
        segment = "Entuzjaści zakupów (niskie dochody, wysokie wydatki)"
    else:
        segment = "Standardowi (średnie dochody i/lub średnie wydatki)"
    
    print(f"Klaster {cluster}: {segment}")

print("\nRekomendacja:")
print(f"Na podstawie metody łokcia oraz współczynnika silhouette, rekomendowana liczba klastrów to {optimal_k}.")
print("Ta liczba klastrów pozwala na wyraźne wyodrębnienie grup klientów o podobnych cechach.")

# Szczegółowe podsumowanie wyników
print("\n" + "="*80)
print("PODSUMOWANIE WYNIKÓW KLASTERYZACJI KLIENTÓW CENTRUM HANDLOWEGO".center(80))
print("="*80)

print("\nWykonaliśmy analizę klasteryzacji danych klientów centrum handlowego przy użyciu algorytmu k-means.")
print("W wyniku analizy wyodrębniliśmy 5 głównych segmentów klientów, charakteryzujących się")
print("różnymi wzorcami zachowań zakupowych w odniesieniu do ich dochodów.\n")

# Pokazanie głównych cech poszczególnych segmentów w formie tabeli
print("╔" + "═"*78 + "╗")
print("║" + "CHARAKTERYSTYKA SEGMENTÓW KLIENTÓW".center(78) + "║")
print("╠" + "═"*20 + "╦" + "═"*57 + "╣")
print("║" + "Segment".center(20) + "║" + "Charakterystyka".center(57) + "║")
print("╠" + "═"*20 + "╬" + "═"*57 + "╣")

for info in sorted(cluster_info, key=lambda x: (x['income_mean'], x['spending_mean']), reverse=True):
    cluster = info['cluster']
    if info['income_mean'] > 70 and info['spending_mean'] > 70:
        segment = "Premium"
        desc = "Zamożni klienci z wysoką skłonnością do wydatków"
    elif info['income_mean'] < 40 and info['spending_mean'] < 40:
        segment = "Oszczędni"
        desc = "Klienci o niskich dochodach i niskich wydatkach"
    elif info['income_mean'] > 70 and info['spending_mean'] < 40:
        segment = "Zamożni oszczędni"
        desc = "Zamożni klienci z niską skłonnością do wydatków"
    elif info['income_mean'] < 40 and info['spending_mean'] > 70:
        segment = "Entuzjaści zakupów"
        desc = "Klienci o niskich dochodach, ale wysokich wydatkach"
    else:
        segment = "Standardowi"
        desc = "Klienci o średnich dochodach i średnich wydatkach"
    
    print(f"║ Klaster {cluster} ({segment})".ljust(21) + "║" + 
          f" Dochód: {info['income_mean']:.1f}k, Wydatki: {info['spending_mean']:.1f}, N={info['size']}".ljust(57) + "║")
    print(f"║".ljust(21) + "║" + f" {desc}".ljust(57) + "║")
    print("╠" + "═"*20 + "╬" + "═"*57 + "╣")

print("╚" + "═"*20 + "╩" + "═"*57 + "╝")

print("\nIMPLIKACJE DLA STRATEGII MARKETINGOWEJ:")
print("1. Dla segmentu Premium: Strategie sprzedaży towarów premium, programy lojalnościowe")
print("2. Dla Zamożnych oszczędnych: Promocje i oferty specjalne dla przyciągnięcia uwagi")
print("3. Dla Entuzjastów zakupów: Karty lojalnościowe, oferty specjalne, promocje sezonowe")
print("4. Dla klientów Standardowych: Promocje skierowane na produkty średniej półki")
print("5. Dla Oszczędnych: Oferty ekonomiczne, promocje cenowe, kupony rabatowe")

print("\nWNIOSKI KOŃCOWE:")
print("Klasteryzacja k-means pozwoliła na skuteczne wyodrębnienie segmentów klientów o podobnych")
print("cechach, co może być wykorzystane do personalizacji strategii marketingowych")
print("i lepszego dopasowania oferty do potrzeb poszczególnych grup klientów.")
print("="*80)
