import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Funkcja do wczytania i przygotowania danych
def wczytaj_dane(plik):
    # Wczytanie danych
    dane = pd.read_csv(plik)
    
    # Informacje o danych
    print("Informacje o zbiorze danych:")
    print(f"Liczba planet: {dane.shape[0]}")
    print(f"Liczba cech: {dane.shape[1]}")
    print("\nPierwsze 5 wierszy:")
    print(dane.head())
    
    # Opis kolumn
    print("\nOpis kolumn:")
    print("pl_name - nazwa planety")
    print("pl_orbper - okres orbitalny (dni)")
    print("pl_orbsmax - wielka półoś orbity (AU)")
    print("pl_rade - promień planety (promienie Ziemi)")
    print("pl_masse - masa planety (masy Ziemi)")
    print("pl_orbeccen - mimośród orbity")
    print("pl_eqt - temperatura równowagowa (K)")
    print("st_teff - temperatura efektywna gwiazdy (K)")
    print("st_mass - masa gwiazdy (masy Słońca)")
    print("sy_dist - odległość układu od Ziemi (parseków)")
    
    # Sprawdzenie brakujących wartości
    print("\nBrakujące wartości:")
    print(dane.isnull().sum())
    
    # Podstawowe statystyki
    print("\nPodstawowe statystyki:")
    print(dane.describe())
    
    return dane

# Funkcja do wykrywania outlierów metodą IQR
def wykryj_outliery_iqr(df, cols, factor=1.5):
    """
    Wykrywa outliery w danych używając metody IQR (Interquartile Range).
    
    Parametry:
    df (DataFrame): DataFrame z danymi
    cols (list): Lista nazw kolumn do analizy
    factor (float): Współczynnik IQR (domyślnie 1.5)
    
    Zwraca:
    list: Indeksy wierszy zawierających outliery
    """
    outlier_indices = []
    
    for col in cols:
        # Pomijamy kolumny z brakującymi wartościami
        if df[col].isnull().any():
            valid_data = df[col].dropna()
        else:
            valid_data = df[col]
        
        Q1 = np.percentile(valid_data, 25)
        Q3 = np.percentile(valid_data, 75)
        IQR = Q3 - Q1
        
        outlier_low = Q1 - factor * IQR
        outlier_high = Q3 + factor * IQR
        
        # Wykrywanie outlierów w kolumnie
        outliers = df[(df[col] < outlier_low) | (df[col] > outlier_high)].index
        outlier_indices.extend(outliers)
    
    # Zwracamy unikalne indeksy
    return list(set(outlier_indices))

# Funkcja do przygotowania danych do klasteryzacji
def przygotuj_dane(dane, cechy, skalowanie='standardowe', usun_outliery=False, iqr_factor=1.5):
    # Wybieramy tylko interesujące nas cechy
    dane_do_klasteryzacji = dane[cechy].copy()
    
    # Usuwanie outlierów jeśli wybrano taką opcję
    if usun_outliery:
        # Najpierw imputujemy brakujące wartości, żeby móc wykryć outliery
        tymczasowy_imputer = SimpleImputer(strategy='median')
        dane_imputowane = pd.DataFrame(
            tymczasowy_imputer.fit_transform(dane_do_klasteryzacji),
            columns=cechy
        )
        
        # Wykrywanie outlierów
        outliery_idx = wykryj_outliery_iqr(dane_imputowane, cechy, iqr_factor)
        
        # Wyświetlenie informacji o outlierach
        if outliery_idx:
            print(f"\nWykryto {len(outliery_idx)} outlierów ({len(outliery_idx)/dane_do_klasteryzacji.shape[0]:.1%} danych)")
            for cecha in cechy:
                if dane[cecha].iloc[outliery_idx].notna().any():
                    min_val = dane[cecha].iloc[outliery_idx].min()
                    max_val = dane[cecha].iloc[outliery_idx].max()
                    print(f"Zakres outlierów dla {cecha}: [{min_val:.2f}, {max_val:.2f}]")
        
        # Usuwanie wierszy z outlierami
        dane_bez_outlierow = dane_do_klasteryzacji.drop(outliery_idx)
        dane_do_klasteryzacji = dane_bez_outlierow
        
        print(f"Liczba danych po usunięciu outlierów: {dane_do_klasteryzacji.shape[0]}")
    
    # Imputacja brakujących wartości (zastępowanie medianą)
    imputer = SimpleImputer(strategy='median')
    dane_do_klasteryzacji = pd.DataFrame(
        imputer.fit_transform(dane_do_klasteryzacji),
        columns=cechy
    )
    
    # Skalowanie danych
    if skalowanie == 'standardowe':
        scaler = StandardScaler()
    elif skalowanie == 'minmax':
        scaler = MinMaxScaler()
    elif skalowanie == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Nieznany typ skalowania")
    
    dane_przeskalowane = scaler.fit_transform(dane_do_klasteryzacji)
    
    return dane_przeskalowane, scaler

# Funkcja do wyznaczenia optymalnej liczby klastrów
def znajdz_optymalna_liczbe_klastrow(dane_przeskalowane, max_klastrow=10):
    # Metoda łokcia (Elbow method)
    plt.figure(figsize=(20, 5))
    
    # Wykres metody łokcia dla KMeans
    plt.subplot(1, 3, 1)
    visualizer = KElbowVisualizer(KMeans(random_state=42), k=(2, max_klastrow+1), metric='distortion', timings=False)
    visualizer.fit(dane_przeskalowane)
    visualizer.finalize()
    plt.title('Metoda łokcia (inercja)', fontsize=14)
    
    # Wykres współczynnika silhouette dla różnych wartości k
    plt.subplot(1, 3, 2)
    silhouette_scores = []
    for k in range(2, max_klastrow+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(dane_przeskalowane)
        score = silhouette_score(dane_przeskalowane, kmeans.labels_)
        silhouette_scores.append(score)
    
    plt.plot(range(2, max_klastrow+1), silhouette_scores, marker='o')
    plt.title('Współczynnik Silhouette', fontsize=14)
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    
    # Wykres współczynnika Calinski-Harabasz dla różnych wartości k
    plt.subplot(1, 3, 3)
    ch_scores = []
    for k in range(2, max_klastrow+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.labels_ = kmeans.fit_predict(dane_przeskalowane)
        score = calinski_harabasz_score(dane_przeskalowane, kmeans.labels_)
        ch_scores.append(score)
    
    plt.plot(range(2, max_klastrow+1), ch_scores, marker='o')
    plt.title('Współczynnik Calinski-Harabasz', fontsize=14)
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Calinski-Harabasz Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Proponowana optymalna liczba klastrów
    silhouette_opt = np.argmax(silhouette_scores) + 2
    ch_opt = np.argmax(ch_scores) + 2
    
    print(f"Proponowana optymalna liczba klastrów:")
    print(f"- Według współczynnika Silhouette: {silhouette_opt}")
    print(f"- Według współczynnika Calinski-Harabasz: {ch_opt}")
    
    return silhouette_opt

def wizualizuj_klastry_2d(dane_przeskalowane, etykiety, centroids=None, tytul="Wizualizacja klastrów"):
    # Redukcja wymiarowości do 2D za pomocą PCA
    pca = PCA(n_components=2)
    dane_pca = pca.fit_transform(dane_przeskalowane)
    
    plt.figure(figsize=(12, 10))
    

    scatter = plt.scatter(dane_pca[:, 0], dane_pca[:, 1], c=etykiety, cmap='viridis', s=50, alpha=0.8)
    
    if centroids is not None:
        centroids_pca = pca.transform(centroids)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='red', edgecolor='k', label='Centroidy')
    
    plt.colorbar(scatter, label='Numer klastra')
    plt.title(tytul, fontsize=15)
    plt.xlabel(f'Pierwsza składowa główna (wyjaśnia {pca.explained_variance_ratio_[0]:.2%} wariancji)', fontsize=12)
    plt.ylabel(f'Druga składowa główna (wyjaśnia {pca.explained_variance_ratio_[1]:.2%} wariancji)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return pca.explained_variance_ratio_

# Funkcja do przeprowadzenia klasteryzacji K-means
def wykonaj_kmeans(dane_przeskalowane, n_clusters):
    # Inicjalizacja i trenowanie modelu K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    etykiety = kmeans.fit_predict(dane_przeskalowane)
    centroids = kmeans.cluster_centers_
    
    # Ocena jakości klasteryzacji
    silhouette = silhouette_score(dane_przeskalowane, etykiety)
    db_score = davies_bouldin_score(dane_przeskalowane, etykiety)
    ch_score = calinski_harabasz_score(dane_przeskalowane, etykiety)
    
    print(f"\nWyniki klasteryzacji K-means (k={n_clusters}):")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Davies-Bouldin Score: {db_score:.3f} (niższe wartości są lepsze)")
    print(f"Calinski-Harabasz Score: {ch_score:.3f}")
    
    # Liczba obiektów w każdym klastrze
    unikalne, liczebnosci = np.unique(etykiety, return_counts=True)
    for i, (klaster, liczba) in enumerate(zip(unikalne, liczebnosci)):
        print(f"Klaster {klaster}: {liczba} planet ({liczba/len(etykiety):.1%})")
    
    return etykiety, centroids, kmeans

# Funkcja do przeprowadzenia klasteryzacji hierarchicznej
def wykonaj_hierarchiczna(dane_przeskalowane, n_clusters):
    # Inicjalizacja i trenowanie modelu klasteryzacji hierarchicznej
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    etykiety = hierarchical.fit_predict(dane_przeskalowane)
    
    # Ocena jakości klasteryzacji
    silhouette = silhouette_score(dane_przeskalowane, etykiety)
    db_score = davies_bouldin_score(dane_przeskalowane, etykiety)
    ch_score = calinski_harabasz_score(dane_przeskalowane, etykiety)
    
    print(f"\nWyniki klasteryzacji hierarchicznej (k={n_clusters}):")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Davies-Bouldin Score: {db_score:.3f} (niższe wartości są lepsze)")
    print(f"Calinski-Harabasz Score: {ch_score:.3f}")
    
    # Liczba obiektów w każdym klastrze
    unikalne, liczebnosci = np.unique(etykiety, return_counts=True)
    for i, (klaster, liczba) in enumerate(zip(unikalne, liczebnosci)):
        print(f"Klaster {klaster}: {liczba} planet ({liczba/len(etykiety):.1%})")
    
    return etykiety

# Funkcja do przeprowadzenia klasteryzacji DBSCAN
def wykonaj_dbscan(dane_przeskalowane, eps=0.5, min_samples=5):
    # Inicjalizacja i trenowanie modelu DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    etykiety = dbscan.fit_predict(dane_przeskalowane)
    
    # Sprawdzenie czy wszystkie punkty zostały przypisane do klastrów
    n_noise = np.sum(etykiety == -1)
    
    if n_noise == len(dane_przeskalowane) or len(np.unique(etykiety)) <= 1:
        print("\nDBSCAN nie znalazł sensownych klastrów z podanymi parametrami.")
        return None
    
    # Ocena jakości klasteryzacji (pomijamy szum)
    idx = etykiety != -1
    if np.sum(idx) > 1 and len(np.unique(etykiety[idx])) > 1:
        silhouette = silhouette_score(dane_przeskalowane[idx], etykiety[idx])
        db_score = davies_bouldin_score(dane_przeskalowane[idx], etykiety[idx])
        ch_score = calinski_harabasz_score(dane_przeskalowane[idx], etykiety[idx])
        
        print(f"\nWyniki klasteryzacji DBSCAN (eps={eps}, min_samples={min_samples}):")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Davies-Bouldin Score: {db_score:.3f} (niższe wartości są lepsze)")
        print(f"Calinski-Harabasz Score: {ch_score:.3f}")
    else:
        print("\nNie można obliczyć metryk jakości dla DBSCAN - zbyt mało klastrów lub punktów niezaszumionych.")
    
    # Liczba obiektów w każdym klastrze
    unikalne, liczebnosci = np.unique(etykiety, return_counts=True)
    for i, (klaster, liczba) in enumerate(zip(unikalne, liczebnosci)):
        if klaster == -1:
            print(f"Szum: {liczba} planet ({liczba/len(etykiety):.1%})")
        else:
            print(f"Klaster {klaster}: {liczba} planet ({liczba/len(etykiety):.1%})")
    
    return etykiety

# Funkcja do analizy i wizualizacji klastrów
def analizuj_klastry(dane, dane_przeskalowane, etykiety, cechy, scaler, metoda="K-means"):
    plt.figure(figsize=(15, 12))
    
    # Dodanie etykiet klastrów do oryginalnych danych
    dane_z_klastrami = dane.copy()
    dane_z_klastrami['klaster'] = etykiety
    
    # Wyliczenie średnich wartości cech dla każdego klastra
    srednie_klastrow = dane_z_klastrami.groupby('klaster')[cechy].mean()
    
    # Wizualizacja średnich wartości cech dla klastrów
    srednie_klastrow_norm = (srednie_klastrow - srednie_klastrow.min()) / (srednie_klastrow.max() - srednie_klastrow.min())
    plt.subplot(2, 1, 1)
    ax = srednie_klastrow_norm.T.plot(kind='bar', figsize=(15, 12), width=0.8)
    plt.title(f'Znormalizowane średnie wartości cech dla klastrów ({metoda})', fontsize=15)
    plt.ylabel('Znormalizowana wartość', fontsize=12)
    plt.xlabel('Cecha', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Klaster')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Rozkład planet w klastrach
    plt.subplot(2, 1, 2)
    dane_z_klastrami['klaster'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title(f'Liczba planet w każdym klastrze ({metoda})', fontsize=15)
    plt.xlabel('Klaster', fontsize=12)
    plt.ylabel('Liczba planet', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Opis charakterystycznych cech klastrów
    print(f"\nCharakterystyki klastrów ({metoda}):")
    for i in range(len(srednie_klastrow)):
        if i in srednie_klastrow.index:
            print(f"\nKlaster {i}:")
            
            # Przekształcenie przeskalowanych wartości z powrotem do oryginalnej skali
            wartosci_klastra = srednie_klastrow.loc[i]
            
            for j, cecha in enumerate(cechy):
                print(f"{cecha}: {wartosci_klastra[cecha]:.2f}")
            
            # Dodatkowy opis klastra
            opis_klastra(wartosci_klastra, cechy)

# Funkcja do opisu klastra na podstawie jego cech
def opis_klastra(wartosci, cechy):
    opis = ""
    
    # Okresy orbitalne
    if 'pl_orbper' in cechy:
        if wartosci['pl_orbper'] < 1:
            opis += "- Ultra krótki okres orbitalny (< 1 dnia)\n"
        elif wartosci['pl_orbper'] < 10:
            opis += "- Krótki okres orbitalny (< 10 dni)\n"
        elif wartosci['pl_orbper'] < 100:
            opis += "- Średni okres orbitalny (10-100 dni)\n"
        else:
            opis += "- Długi okres orbitalny (> 100 dni)\n"
    
    # Promień
    if 'pl_rade' in cechy:
        if wartosci['pl_rade'] < 1.5:
            opis += "- Planeta typu ziemskiego (R < 1.5 R_Ziemi)\n"
        elif wartosci['pl_rade'] < 4:
            opis += "- Super-Ziemia/Mini-Neptun (1.5 < R < 4 R_Ziemi)\n"
        elif wartosci['pl_rade'] < 10:
            opis += "- Planeta typu Neptuna (4 < R < 10 R_Ziemi)\n"
        else:
            opis += "- Planeta typu gazowego olbrzyma (R > 10 R_Ziemi)\n"
    
    # Masa
    if 'pl_masse' in cechy:
        if wartosci['pl_masse'] < 10:
            opis += "- Mała masa (< 10 mas Ziemi)\n"
        elif wartosci['pl_masse'] < 100:
            opis += "- Średnia masa (10-100 mas Ziemi)\n"
        elif wartosci['pl_masse'] < 1000:
            opis += "- Duża masa (100-1000 mas Ziemi)\n"
        else:
            opis += "- Bardzo duża masa (> 1000 mas Ziemi)\n"
    
    # Temperatura równowagowa
    if 'pl_eqt' in cechy:
        if wartosci['pl_eqt'] < 500:
            opis += "- Chłodna planeta (< 500K)\n"
        elif wartosci['pl_eqt'] < 1000:
            opis += "- Umiarkowanie ciepła planeta (500-1000K)\n"
        elif wartosci['pl_eqt'] < 1500:
            opis += "- Gorąca planeta (1000-1500K)\n"
        else:
            opis += "- Bardzo gorąca planeta (> 1500K)\n"
    
    # Mimośród orbity
    if 'pl_orbeccen' in cechy:
        if wartosci['pl_orbeccen'] < 0.1:
            opis += "- Prawie kolista orbita (e < 0.1)\n"
        elif wartosci['pl_orbeccen'] < 0.3:
            opis += "- Umiarkowanie eliptyczna orbita (0.1 < e < 0.3)\n"
        else:
            opis += "- Wysoce eliptyczna orbita (e > 0.3)\n"
    
    print(opis)

    # Funkcja do porównania wyników różnych metod klasteryzacji
    def porownaj_metody(dane_przeskalowane, etykiety_kmeans, etykiety_hierarchical, etykiety_dbscan=None):
        plt.figure(figsize=(15, 5))
        
        # Redukcja wymiarowości do 2D za pomocą PCA
        pca = PCA(n_components=2)
        dane_pca = pca.fit_transform(dane_przeskalowane)
        
        # Wizualizacja wyników K-means
        plt.subplot(1, 3, 1)
        plt.scatter(dane_pca[:, 0], dane_pca[:, 1], c=etykiety_kmeans, cmap='viridis', s=40, alpha=0.8)
        plt.title('K-means')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(label='Klaster')
        
        # Wizualizacja wyników klasteryzacji hierarchicznej
        plt.subplot(1, 3, 2)
        plt.scatter(dane_pca[:, 0], dane_pca[:, 1], c=etykiety_hierarchical, cmap='viridis', s=40, alpha=0.8)
        plt.title('Klasteryzacja hierarchiczna')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(label='Klaster')
        
        # Wizualizacja wyników DBSCAN, jeśli dostępne
        if etykiety_dbscan is not None:
            plt.subplot(1, 3, 3)
            plt.scatter(dane_pca[:, 0], dane_pca[:, 1], c=etykiety_dbscan, cmap='viridis', s=40, alpha=0.8)
            plt.title('DBSCAN')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.colorbar(label='Klaster')
        
        plt.tight_layout()
        plt.show()

# Główna funkcja wykonująca analizę klasteryzacji
def analiza_klasteryzacji_planet(sciezka_danych='../../data/lab6/planets.csv', usun_outliery=False):
    # Wczytanie danych
    dane = wczytaj_dane(sciezka_danych)
    
    # Wybór cech do klasteryzacji
    cechy = ['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_masse', 'pl_orbeccen', 'pl_eqt', 'st_teff', 'st_mass']
    
    # Przygotowanie danych
    dane_przeskalowane, scaler = przygotuj_dane(dane, cechy, usun_outliery=usun_outliery)
    
    # Znalezienie optymalnej liczby klastrów
    optymalna_liczba_klastrow = znajdz_optymalna_liczbe_klastrow(dane_przeskalowane)
    
    # Przeprowadzenie klasteryzacji K-means
    etykiety_kmeans, centroids_kmeans, model_kmeans = wykonaj_kmeans(dane_przeskalowane, optymalna_liczba_klastrow)
    
    # Wizualizacja klastrów w przestrzeni 2D
    wizualizuj_klastry_2d(dane_przeskalowane, etykiety_kmeans, centroids_kmeans, "Klasteryzacja K-means")
    
    # Przeprowadzenie klasteryzacji hierarchicznej
    etykiety_hierarchical = wykonaj_hierarchiczna(dane_przeskalowane, optymalna_liczba_klastrow)
    
    # Wizualizacja klastrów z klasteryzacji hierarchicznej
    wizualizuj_klastry_2d(dane_przeskalowane, etykiety_hierarchical, tytul="Klasteryzacja hierarchiczna")
    
    # Przeprowadzenie klasteryzacji DBSCAN z dobranymi parametrami
    etykiety_dbscan = wykonaj_dbscan(dane_przeskalowane, eps=0.8, min_samples=5)
    
    # Wizualizacja klastrów z DBSCAN, jeśli klasteryzacja się powiodła
    if etykiety_dbscan is not None and len(np.unique(etykiety_dbscan)) > 1:
        wizualizuj_klastry_2d(dane_przeskalowane, etykiety_dbscan, tytul="Klasteryzacja DBSCAN")
    
    # Porównanie wyników różnych metod klasteryzacji
    porownaj_metody(dane_przeskalowane, etykiety_kmeans, etykiety_hierarchical, etykiety_dbscan)
    
    # Analiza klastrów K-means
    analizuj_klastry(dane, dane_przeskalowane, etykiety_kmeans, cechy, scaler, "K-means")
    
    # Analiza klastrów klasteryzacji hierarchicznej
    analizuj_klastry(dane, dane_przeskalowane, etykiety_hierarchical, cechy, scaler, "Hierarchiczna")
    
    print("\n=== WNIOSKI ===")
    print("1. Porównanie wyników różnych metod klasteryzacji pokazuje, że...")
    print("2. Najlepsze rezultaty uzyskano przy użyciu...")
    print("3. Charakterystyczne cechy klastrów wskazują na...")
    print("4. Optymalna liczba klastrów wynosi...")

# Funkcja do przygotowania i uruchomienia w Google Colab
def uruchom_w_colab():
    """
    Funkcja do uruchomienia analizy w Google Colab.
    
    Aby uruchomić ten skrypt w Google Colab:
    1. Załaduj plik planets.csv do Colab (lub pobierz go bezpośrednio)
    2. Uruchom poniższy kod:
    
    ```python
    !pip install yellowbrick
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    import seaborn as sns
    from yellowbrick.cluster import KElbowVisualizer
    from scipy.cluster.hierarchy import dendrogram, linkage
    import warnings
    warnings.filterwarnings('ignore')
    
    # Pobierz plik planets.csv (opcjonalnie)
    !wget https://raw.githubusercontent.com/username/repo/main/planets.csv -O planets.csv
    
    # Skopiuj i wklej cały kod z pliku planets.py poniżej
    
    # Uruchom analizę
    analiza_klasteryzacji_planet('planets.csv')
    ```
    """
    print("Ta funkcja służy jako instrukcja do uruchomienia skryptu w Google Colab.")
    print("Aby uruchomić analizę w lokalnym środowisku, użyj funkcji analiza_klasteryzacji_planet().")

# Wywołanie głównej funkcji
if __name__ == "__main__":
    print("\n=== ANALIZA Z OUTLIERAMI ===")
    analiza_klasteryzacji_planet()
    
    print("\n\n=== ANALIZA BEZ OUTLIERÓW ===")
    analiza_klasteryzacji_planet(usun_outliery=True)
    
    # Wydrukuj instrukcję dla Google Colab
    print("\n=== INSTRUKCJA DLA GOOGLE COLAB ===")
    print("Aby uruchomić ten skrypt w Google Colab, załaduj plik planets.csv i użyj funkcji analiza_klasteryzacji_planet('planets.csv')")
    print("Alternatywnie, możesz skopiować cały kod tego pliku do notebooka Colab i uruchomić go bezpośrednio.") 