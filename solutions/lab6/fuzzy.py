import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def wczytaj_dane(plik):
    """
    Wczytuje i wyświetla podstawowe informacje o zbiorze danych planet.
    
    Args:
        plik: Ścieżka do pliku CSV z danymi planetarnymi.
        
    Returns:
        DataFrame zawierający wczytane dane.
    """
    dane = pd.read_csv(plik)
    
    print("Informacje o zbiorze danych:")
    print(f"Liczba planet: {dane.shape[0]}")
    print(f"Liczba cech: {dane.shape[1]}")
    print("\nPierwsze 5 wierszy:")
    print(dane.head())
    
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
    
    print("\nBrakujące wartości:")
    print(dane.isnull().sum())
    
    print("\nPodstawowe statystyki:")
    print(dane.describe())
    
    return dane

def przygotuj_dane(dane, cechy, skalowanie='standardowe'):
    """
    Przygotowuje dane do klasteryzacji: imputuje brakujące wartości i skaluje dane.
    
    Args:
        dane: DataFrame z danymi.
        cechy: Lista nazw kolumn do wykorzystania.
        skalowanie: Metoda skalowania danych ('standardowe' lub 'minmax').
        
    Returns:
        dane_przeskalowane: Numpy array z przeskalowanymi danymi.
        scaler: Obiekt skalera.
    """
    dane_do_klasteryzacji = dane[cechy].copy()
    
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
    else:
        raise ValueError("Nieznany typ skalowania")
    
    dane_przeskalowane = scaler.fit_transform(dane_do_klasteryzacji)
    
    return dane_przeskalowane, scaler

def znajdz_optymalny_c(dane_przeskalowane, max_c=10):
    """
    Znajduje optymalną liczbę klastrów dla algorytmu fuzzy c-means.
    
    Args:
        dane_przeskalowane: Numpy array z przeskalowanymi danymi.
        max_c: Maksymalna liczba klastrów do sprawdzenia.
        
    Returns:
        int: Optymalna liczba klastrów.
    """
    fpcs = []
    
    # Uwaga: dane muszą być w formacie (cechy, obiekty) dla skfuzzy.cmeans
    # dane_fcm to dane w formacie (cechy, obiekty)
    dane_fcm = dane_przeskalowane.T
    
    # Sprawdzanie różnych wartości c (liczby klastrów)
    for c in range(2, max_c + 1):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            dane_fcm, c, 2, error=0.005, maxiter=1000, init=None
        )
        fpcs.append(fpc)
    
    # Wizualizacja współczynników FPC
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_c + 1), fpcs, 'bo-')
    plt.title('Współczynnik rozmytej podzielności (FPC) dla różnych wartości c')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Współczynnik FPC')
    plt.grid(True)
    plt.show()
    
    # Wybór optymalnej wartości c (maksymalizacja FPC)
    optimal_c = np.argmax(fpcs) + 2
    print(f"Optymalna liczba klastrów według współczynnika FPC: {optimal_c}")
    
    return optimal_c

def wykonaj_fcm(dane_przeskalowane, n_clusters, m=2):
    """
    Wykonuje klasteryzację metodą Fuzzy C-Means.
    
    Args:
        dane_przeskalowane: Numpy array z przeskalowanymi danymi.
        n_clusters: Liczba klastrów.
        m: Parametr rozmycia (m > 1).
        
    Returns:
        cntr: Centroidy klastrów.
        u: Macierz przynależności.
        fpc: Współczynnik rozmytej podzielności.
    """
    # Dane w formacie (cechy, obiekty) dla skfuzzy.cmeans
    dane_fcm = dane_przeskalowane.T
    
    # Wykonanie klasteryzacji FCM
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        dane_fcm, n_clusters, m, error=0.005, maxiter=1000, init=None
    )
    
    print(f"\nWyniki klasteryzacji Fuzzy C-Means (c={n_clusters}, m={m}):")
    print(f"Współczynnik rozmytej podzielności (FPC): {fpc:.3f}")
    print("(FPC bliższy 1 oznacza lepszą klasteryzację)")
    
    # Przypisanie do klastrów (na podstawie najwyższego stopnia przynależności)
    etykiety = np.argmax(u, axis=0)
    
    # Liczba obiektów w każdym klastrze (na podstawie najwyższego stopnia przynależności)
    unikalne, liczebnosci = np.unique(etykiety, return_counts=True)
    for i, (klaster, liczba) in enumerate(zip(unikalne, liczebnosci)):
        print(f"Klaster {klaster}: {liczba} planet ({liczba/len(etykiety):.1%})")
    
    return cntr, u, fpc, etykiety

def wizualizuj_klastry_2d(dane_przeskalowane, u, etykiety, centroids, tytul="Wizualizacja klastrów FCM"):
    """
    Wizualizuje wyniki klasteryzacji w przestrzeni 2D (PCA).
    
    Args:
        dane_przeskalowane: Numpy array z przeskalowanymi danymi.
        u: Macierz przynależności.
        etykiety: Etykiety klastrów.
        centroids: Centroidy klastrów.
        tytul: Tytuł wykresu.
    """
    # Redukcja wymiarowości do 2D za pomocą PCA
    pca = PCA(n_components=2)
    dane_pca = pca.fit_transform(dane_przeskalowane)
    
    # Wizualizacja klastrów
    plt.figure(figsize=(12, 10))
    
    # Wykres punktów z kolorami według klastra
    scatter = plt.scatter(dane_pca[:, 0], dane_pca[:, 1], c=etykiety, cmap='viridis', s=50, alpha=0.8)
    
    # Dodanie centroidów - centroids mają format (klastry, cechy), więc nie transpozycjonujemy dalej
    centroids_pca = pca.transform(centroids)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='red', edgecolor='k', label='Centroidy')
    
    plt.colorbar(scatter, label='Numer klastra')
    plt.title(tytul, fontsize=15)
    plt.xlabel(f'Pierwsza składowa główna (wyjaśnia {pca.explained_variance_ratio_[0]:.2%} wariancji)', fontsize=12)
    plt.ylabel(f'Druga składowa główna (wyjaśnia {pca.explained_variance_ratio_[1]:.2%} wariancji)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # Wizualizacja stopni przynależności
    plt.figure(figsize=(12, 8))
    for i in range(u.shape[0]):
        plt.plot(u[i], '.', label=f'Klaster {i}')
    
    plt.title('Stopnie przynależności punktów do klastrów', fontsize=15)
    plt.xlabel('Punkty danych (planety)', fontsize=12)
    plt.ylabel('Stopień przynależności', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return pca.explained_variance_ratio_

def analizuj_klastry_fcm(dane, dane_przeskalowane, etykiety, u, cechy, scaler):
    """
    Analizuje i wizualizuje klastry utworzone przez algorytm Fuzzy C-Means.
    
    Args:
        dane: Oryginalny DataFrame z danymi.
        dane_przeskalowane: Numpy array z przeskalowanymi danymi.
        etykiety: Etykiety klastrów.
        u: Macierz przynależności.
        cechy: Lista nazw kolumn.
        scaler: Obiekt skalera.
    """
    plt.figure(figsize=(15, 12))
    
    # Dodanie etykiet klastrów do oryginalnych danych
    dane_z_klastrami = dane.copy()
    dane_z_klastrami['klaster'] = etykiety
    
    # Wizualizacja macierzy przynależności jako mapy ciepła
    plt.subplot(2, 1, 1)
    ax = sns.heatmap(u, cmap='viridis', annot=False)
    plt.title('Macierz przynależności FCM', fontsize=15)
    plt.xlabel('Punkty danych (planety)', fontsize=12)
    plt.ylabel('Klastry', fontsize=12)
    
    # Wyliczenie średnich wartości cech dla każdego klastra
    srednie_klastrow = dane_z_klastrami.groupby('klaster')[cechy].mean()
    
    # Wizualizacja średnich wartości cech dla klastrów
    srednie_klastrow_norm = (srednie_klastrow - srednie_klastrow.min()) / (srednie_klastrow.max() - srednie_klastrow.min())
    plt.subplot(2, 1, 2)
    ax = srednie_klastrow_norm.T.plot(kind='bar', figsize=(15, 12), width=0.8)
    plt.title('Znormalizowane średnie wartości cech dla klastrów (FCM)', fontsize=15)
    plt.ylabel('Znormalizowana wartość', fontsize=12)
    plt.xlabel('Cecha', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Klaster')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Opis charakterystycznych cech klastrów
    print(f"\nCharakterystyki klastrów (FCM):")
    for i in range(len(srednie_klastrow)):
        if i in srednie_klastrow.index:
            print(f"\nKlaster {i}:")
            
            # Przekształcenie przeskalowanych wartości z powrotem do oryginalnej skali
            wartosci_klastra = srednie_klastrow.loc[i]
            
            for j, cecha in enumerate(cechy):
                print(f"{cecha}: {wartosci_klastra[cecha]:.2f}")
            
            # Dodatkowy opis klastra
            opis_klastra(wartosci_klastra, cechy)
            
            # Informacje o stopniach przynależności
            avg_membership = np.mean(u[i][etykiety == i])
            max_membership = np.max(u[i][etykiety == i])
            min_membership = np.min(u[i][etykiety == i])
            print(f"Średni stopień przynależności: {avg_membership:.2f}")
            print(f"Zakres stopni przynależności: [{min_membership:.2f}, {max_membership:.2f}]")

def opis_klastra(wartosci, cechy):
    """
    Generuje opis klastra na podstawie jego cech.
    
    Args:
        wartosci: Wartości cech klastra.
        cechy: Lista nazw cech.
    """
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

def analiza_fcm(sciezka_danych='planets.csv', m=2):
    """
    Główna funkcja wykonująca analizę klasteryzacji planet metodą Fuzzy C-Means.
    
    Args:
        sciezka_danych: Ścieżka do pliku CSV z danymi.
        m: Parametr rozmycia (m > 1).
    """
    # Wczytanie danych
    dane = wczytaj_dane(sciezka_danych)
    
    # Wybór połowy cech do klasteryzacji
    cechy = ['pl_orbper', 'pl_rade', 'pl_masse', 'pl_eqt', 'st_mass']
    
    print(f"\nWybrane cechy do klasteryzacji ({len(cechy)} z {dane.shape[1]}):")
    for cecha in cechy:
        print(f"- {cecha}")
    
    # Przygotowanie danych
    dane_przeskalowane, scaler = przygotuj_dane(dane, cechy)
    
    # Znalezienie optymalnej liczby klastrów
    optymalna_liczba_klastrow = znajdz_optymalny_c(dane_przeskalowane)
    
    # Wykonanie klasteryzacji FCM
    centroids, u, fpc, etykiety = wykonaj_fcm(dane_przeskalowane, optymalna_liczba_klastrow, m)
    
    # Wizualizacja klastrów
    wizualizuj_klastry_2d(dane_przeskalowane, u, etykiety, centroids, "Klasteryzacja Fuzzy C-Means")
    
    # Analiza klastrów
    analizuj_klastry_fcm(dane, dane_przeskalowane, etykiety, u, cechy, scaler)
    
    print("\n=== WNIOSKI ===")
    print("1. Algorytm Fuzzy C-Means pozwala na elastyczną klasyfikację planet, gdzie każda planeta może należeć do wielu klastrów.")
    print("2. Stopnie przynależności dają lepszy wgląd w charakterystykę obiektów na granicy klastrów.")
    print(f"3. Optymalna liczba klastrów wynosi {optymalna_liczba_klastrow} według współczynnika FPC.")
    
    return dane, dane_przeskalowane, etykiety, u, centroids, cechy

# Kod do uruchomienia w Google Colab
"""
Aby uruchomić ten skrypt w Google Colab, użyj poniższego kodu:

!pip install scikit-fuzzy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import skfuzzy as fuzz
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Pobierz dane planetarne
!wget https://raw.githubusercontent.com/abrogow/SciCompPy/master/data/lab6/planets.csv

# Definicja funkcji (wklej cały kod fuzzy.py)
# ...

# Uruchom analizę
dane, dane_przeskalowane, etykiety, u, centroids, cechy = analiza_fcm('planets.csv')
"""

# Wywołanie głównej funkcji
if __name__ == "__main__":
    print("\n=== ANALIZA FUZZY C-MEANS ===")
    try:
        dane, dane_przeskalowane, etykiety, u, centroids, cechy = analiza_fcm()
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        print("\nSpróbuj uruchomić kod w Google Colab używając instrukcji powyżej.")
