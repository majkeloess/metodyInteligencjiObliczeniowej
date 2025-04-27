import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import skfuzzy as fuzz
import warnings
warnings.filterwarnings('ignore')

def wczytaj_dane(plik):
    """
    Funkcja wczytująca dane z pliku CSV i przygotowująca je do analizy.
    """
    dane = pd.read_csv(plik)
    
    print("Informacje o zbiorze danych:")
    print(f"Liczba planet: {dane.shape[0]}")
    print(f"Liczba cech: {dane.shape[1]}")
    print("\nPierwsze 5 wierszy:")
    print(dane.head())
    
    # Sprawdzenie brakujących wartości
    print("\nBrakujące wartości:")
    print(dane.isnull().sum())
    
    return dane

def przygotuj_dane(dane, cechy):
    """
    Funkcja przygotowująca dane do klasteryzacji, obejmująca imputację i standaryzację.
    """
    # Wybieramy tylko interesujące nas cechy (połowę dostępnych kolumn)
    dane_do_klasteryzacji = dane[cechy].copy()
    
    # Imputacja brakujących wartości (zastępowanie medianą)
    imputer = SimpleImputer(strategy='median')
    dane_do_klasteryzacji = pd.DataFrame(
        imputer.fit_transform(dane_do_klasteryzacji),
        columns=cechy
    )
    
    # Standaryzacja danych
    scaler = StandardScaler()
    dane_przeskalowane = scaler.fit_transform(dane_do_klasteryzacji)
    
    return dane_przeskalowane, scaler, dane_do_klasteryzacji

def znajdz_optymalna_liczbe_klastrow_fcm(dane_przeskalowane, max_klastrow=10):
    """
    Funkcja do znalezienia optymalnej liczby klastrów dla fuzzy c-means.
    Wykorzystuje wskaźnik Fukuyama-Sugeno i Xie-Beni.
    """
    fs_scores = []  # Fukuyama-Sugeno Index
    xb_scores = []  # Xie-Beni Index
    
    # Obliczamy wskaźniki jakości dla różnych liczb klastrów
    for n_clusters in range(2, max_klastrow+1):
        # Fuzzy c-means
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            dane_przeskalowane.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
        )
        
        # Fukuyama-Sugeno Index (niższe wartości są lepsze)
        fs_score = fuzz.cluster.cmeans_metrics.fukuyama_sugeno_index(dane_przeskalowane.T, u, cntr)
        fs_scores.append(fs_score)
        
        # Xie-Beni Index (niższe wartości są lepsze)
        xb_score = fuzz.cluster.cmeans_metrics.xie_beni_index(dane_przeskalowane.T, u, cntr)
        xb_scores.append(xb_score)
    
    # Wizualizacja wskaźników
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_klastrow+1), fs_scores, marker='o')
    plt.title('Wskaźnik Fukuyama-Sugeno', fontsize=14)
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Fukuyama-Sugeno Index')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_klastrow+1), xb_scores, marker='o')
    plt.title('Wskaźnik Xie-Beni', fontsize=14)
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Xie-Beni Index')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Optymalna liczba klastrów to minimum dla wskaźników
    fs_opt = np.argmin(fs_scores) + 2  # +2 bo zaczynamy od 2 klastrów
    xb_opt = np.argmin(xb_scores) + 2
    
    print(f"Proponowana optymalna liczba klastrów:")
    print(f"- Według wskaźnika Fukuyama-Sugeno: {fs_opt}")
    print(f"- Według wskaźnika Xie-Beni: {xb_opt}")
    
    # Zwracamy optymalną liczbę klastrów jako średnią z obu metod (zaokrągloną w dół)
    return max(fs_opt, xb_opt)

def wykonaj_fuzzy_cmeans(dane_przeskalowane, n_clusters, m=2):
    """
    Wykonuje klasteryzację metodą fuzzy c-means.
    
    Parametry:
    - dane_przeskalowane: znormalizowane dane
    - n_clusters: liczba klastrów
    - m: parametr rozmycia (zazwyczaj między 1.5 a 2.5)
    
    Zwraca:
    - cntr: centra klastrów
    - u: macierz przynależności
    - fpc: współczynnik partycji (miara jakości klasteryzacji)
    """
    # Transformacja danych do wymaganego formatu (transponowanie)
    dane_t = dane_przeskalowane.T
    
    # Wykonanie klasteryzacji fuzzy c-means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        dane_t, n_clusters, m, error=0.005, maxiter=1000, init=None
    )
    
    print(f"\nWyniki klasteryzacji Fuzzy C-Means (k={n_clusters}, m={m}):")
    print(f"Współczynnik partycji: {fpc:.3f} (wyższe wartości oznaczają lepszą klasteryzację)")
    
    # Obliczamy wskaźniki jakości
    fs_index = fuzz.cluster.cmeans_metrics.fukuyama_sugeno_index(dane_t, u, cntr)
    xb_index = fuzz.cluster.cmeans_metrics.xie_beni_index(dane_t, u, cntr)
    
    print(f"Wskaźnik Fukuyama-Sugeno: {fs_index:.3f} (niższe wartości są lepsze)")
    print(f"Wskaźnik Xie-Beni: {xb_index:.3f} (niższe wartości są lepsze)")
    
    return cntr, u, fpc

def przypisz_planety_do_klastrow(u, prog_przynaleznosci=0.5):
    """
    Przypisuje planety do klastrów na podstawie macierzy przynależności.
    Zwraca etykiety klastrów oraz informacje o planetach z niskim stopniem przynależności.
    """
    # Przypisanie do klastra z najwyższym stopniem przynależności
    etykiety = np.argmax(u, axis=0)
    
    # Sprawdzenie, które planety mają niski stopień przynależności do głównego klastra
    max_przynaleznosc = np.max(u, axis=0)
    niepewne_planety = np.where(max_przynaleznosc < prog_przynaleznosci)[0]
    
    print(f"\nLiczba planet z niskim stopniem przynależności (<{prog_przynaleznosci}): {len(niepewne_planety)}")
    
    # Informacje o liczbie planet w każdym klastrze
    unikalne, liczebnosci = np.unique(etykiety, return_counts=True)
    for i, (klaster, liczba) in enumerate(zip(unikalne, liczebnosci)):
        print(f"Klaster {klaster}: {liczba} planet ({liczba/len(etykiety):.1%})")
    
    return etykiety, niepewne_planety

def wizualizuj_fuzzy_klastry_2d(dane_przeskalowane, u, cntr, tytul="Wizualizacja klastrów Fuzzy C-Means"):
    """
    Wizualizuje wyniki klasteryzacji fuzzy c-means w przestrzeni 2D po redukcji wymiarowości.
    """
    # Redukcja wymiarowości do 2D za pomocą PCA
    pca = PCA(n_components=2)
    dane_pca = pca.fit_transform(dane_przeskalowane)
    
    # Przekształcamy centra klastrów do przestrzeni PCA
    cntr_pca = pca.transform(cntr.T)
    
    # Przypisanie do klastra z najwyższym stopniem przynależności
    etykiety = np.argmax(u, axis=0)
    
    # Wizualizacja klastrów
    plt.figure(figsize=(15, 10))
    
    # Wykres punktów z kolorami według klastra
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(dane_pca[:, 0], dane_pca[:, 1], c=etykiety, cmap='viridis', s=30, alpha=0.8)
    plt.scatter(cntr_pca[:, 0], cntr_pca[:, 1], marker='*', s=500, c='red', edgecolor='k', label='Centra klastrów')
    plt.colorbar(scatter, label='Klaster')
    plt.title(f"{tytul} - Przypisanie do głównego klastra", fontsize=15)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} wariancji)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} wariancji)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Wykres stopnia przynależności
    plt.subplot(1, 2, 2)
    max_przynaleznosc = np.max(u, axis=0)
    scatter = plt.scatter(dane_pca[:, 0], dane_pca[:, 1], c=max_przynaleznosc, cmap='coolwarm', s=30, alpha=0.8)
    plt.colorbar(scatter, label='Stopień przynależności do głównego klastra')
    plt.title(f"{tytul} - Stopień przynależności", fontsize=15)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} wariancji)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} wariancji)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return pca.explained_variance_ratio_

def analizuj_fuzzy_klastry(dane, dane_przeskalowane, u, cechy, scaler):
    """
    Analizuje charakterystyki klastrów fuzzy c-means.
    """
    # Przypisanie do klastra z najwyższym stopniem przynależności
    etykiety = np.argmax(u, axis=0)
    
    # Dodanie etykiet klastrów do oryginalnych danych
    dane_z_klastrami = dane.copy()
    dane_z_klastrami['klaster'] = etykiety
    dane_z_klastrami['stopien_przynaleznosci'] = np.max(u, axis=0)
    
    # Wyliczenie średnich wartości cech dla każdego klastra (tylko dla wybranych cech)
    srednie_klastrow = dane_z_klastrami.groupby('klaster')[cechy].mean()
    
    plt.figure(figsize=(15, 10))
    
    # Wizualizacja średnich wartości cech dla klastrów
    srednie_klastrow_norm = (srednie_klastrow - srednie_klastrow.min()) / (srednie_klastrow.max() - srednie_klastrow.min())
    plt.subplot(2, 1, 1)
    ax = srednie_klastrow_norm.T.plot(kind='bar', figsize=(15, 10), width=0.8)
    plt.title('Znormalizowane średnie wartości cech dla klastrów (Fuzzy C-Means)', fontsize=15)
    plt.ylabel('Znormalizowana wartość', fontsize=12)
    plt.xlabel('Cecha', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Klaster')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Rozkład planet w klastrach
    plt.subplot(2, 1, 2)
    dane_z_klastrami['klaster'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title('Liczba planet w każdym klastrze (Fuzzy C-Means)', fontsize=15)
    plt.xlabel('Klaster', fontsize=12)
    plt.ylabel('Liczba planet', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Opis charakterystycznych cech klastrów
    print("\nCharakterystyki klastrów (Fuzzy C-Means):")
    for i in range(len(srednie_klastrow)):
        if i in srednie_klastrow.index:
            print(f"\nKlaster {i}:")
            wartosci_klastra = srednie_klastrow.loc[i]
            
            for cecha in cechy:
                print(f"{cecha}: {wartosci_klastra[cecha]:.2f}")
            
            # Dodatkowy opis klastra
            opis_klastra(wartosci_klastra, cechy)

def opis_klastra(wartosci, cechy):
    """
    Generuje opis klastra na podstawie jego cech.
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

def porownaj_czlonkostwo_klastrow(u, prog_przynaleznosci=0.5):
    """
    Analizuje rozkład stopni przynależności planet do klastrów.
    """
    plt.figure(figsize=(15, 6))
    
    # Histogram maksymalnych stopni przynależności
    plt.subplot(1, 2, 1)
    max_przynaleznosc = np.max(u, axis=0)
    plt.hist(max_przynaleznosc, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=prog_przynaleznosci, color='red', linestyle='--', label=f'Próg: {prog_przynaleznosci}')
    plt.title('Histogram maksymalnych stopni przynależności', fontsize=15)
    plt.xlabel('Stopień przynależności do głównego klastra', fontsize=12)
    plt.ylabel('Liczba planet', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Entropia przynależności (miara niepewności przypisania)
    plt.subplot(1, 2, 2)
    # Normalizacja macierzy przynależności do zakresu [0,1]
    u_norm = u / u.sum(axis=0)
    # Obliczenie entropii dla każdej planety
    entropia = -np.sum(u_norm * np.log2(u_norm + 1e-10), axis=0)
    # Maksymalna możliwa entropia dla danej liczby klastrów
    max_entropia = np.log2(u.shape[0])
    # Normalizacja entropii
    entropia_norm = entropia / max_entropia
    
    plt.hist(entropia_norm, bins=20, color='lightgreen', edgecolor='black')
    plt.title('Histogram entropii przynależności', fontsize=15)
    plt.xlabel('Znormalizowana entropia przynależności', fontsize=12)
    plt.ylabel('Liczba planet', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Średni stopień przynależności do głównego klastra: {np.mean(max_przynaleznosc):.3f}")
    print(f"Mediana stopnia przynależności: {np.median(max_przynaleznosc):.3f}")
    print(f"Liczba planet z niejednoznacznym przypisaniem (< {prog_przynaleznosci}): {np.sum(max_przynaleznosc < prog_przynaleznosci)}")
    print(f"Średnia znormalizowana entropia przynależności: {np.mean(entropia_norm):.3f} (0=pewne przypisanie, 1=maksymalna niepewność)")

def analiza_fuzzy_clustering_planet(sciezka_danych='../../data/lab6/planets.csv'):
    """
    Główna funkcja wykonująca analizę klasteryzacji metodą fuzzy c-means.
    """
    # Wczytanie danych
    dane = wczytaj_dane(sciezka_danych)
    
    # Wybór połowy dostępnych cech (5 z 10 kolumn, pomijając nazwę planety)
    dostepne_cechy = ['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_masse', 'pl_orbeccen', 
                     'pl_eqt', 'st_teff', 'st_mass', 'sy_dist']
    
    # Wybieramy tylko połowę cech
    cechy = ['pl_orbper', 'pl_rade', 'pl_masse', 'pl_eqt', 'st_teff']
    
    print(f"\nWybrane cechy do analizy ({len(cechy)} z {len(dostepne_cechy)}):")
    for cecha in cechy:
        print(f"- {cecha}")
    
    # Przygotowanie danych
    dane_przeskalowane, scaler, dane_oryginalne = przygotuj_dane(dane, cechy)
    
    # Znalezienie optymalnej liczby klastrów
    optymalna_liczba_klastrow = znajdz_optymalna_liczbe_klastrow_fcm(dane_przeskalowane)
    
    # Parametr rozmycia m (typowo między 1.5 a 2.5)
    m = 2.0
    
    # Wykonanie klasteryzacji fuzzy c-means
    cntr, u, fpc = wykonaj_fuzzy_cmeans(dane_przeskalowane, optymalna_liczba_klastrow, m)
    
    # Przypisanie planet do klastrów
    etykiety, niepewne_planety = przypisz_planety_do_klastrow(u)
    
    # Wizualizacja klastrów
    wizualizuj_fuzzy_klastry_2d(dane_przeskalowane, u, cntr, "Klasteryzacja Fuzzy C-Means")
    
    # Analiza rozkładu stopni przynależności
    porownaj_czlonkostwo_klastrow(u)
    
    # Analiza charakterystyk klastrów
    analizuj_fuzzy_klastry(dane, dane_przeskalowane, u, cechy, scaler)
    
    print("\n=== WNIOSKI Z FUZZY CLUSTERINGU ===")
    print(f"1. Optymalna liczba klastrów dla danych egzoplanet to {optymalna_liczba_klastrow}.")
    print(f"2. Współczynnik partycji wyniósł {fpc:.3f}, co wskazuje na...")
    print("3. Klasteryzacja rozmyta pozwoliła zidentyfikować planety o niejednoznacznej przynależności do klastrów.")
    print("4. Główne grupy egzoplanet to: [tu należy wpisać charakterystykę klastrów na podstawie analizy]")
    print("5. Zalety zastosowania fuzzy clusteringu w porównaniu do tradycyjnych metod:")
    print("   - Możliwość identyfikacji obiektów granicznych")
    print("   - Bardziej realistyczne odzwierciedlenie ciągłej natury różnic między egzoplanetami")
    print("   - Lepsze radzenie sobie z szumem i niepewnością w danych")

def uruchom_w_colab():
    """
    Funkcja do uruchomienia analizy w Google Colab.
    """
    print("""
    Aby uruchomić analizę fuzzy clusteringu w Google Colab:
    
    1. Zainstaluj wymagane biblioteki:
    
    ```python
    !pip install scikit-fuzzy yellowbrick
    ```
    
    2. Wczytaj dane:
    
    ```python
    !wget https://raw.githubusercontent.com/username/repo/main/planets.csv -O planets.csv
    ```
    
    3. Skopiuj cały kod z pliku fuzzy_planets.py
    
    4. Uruchom analizę:
    
    ```python
    analiza_fuzzy_clustering_planet('planets.csv')
    ```
    """)

# Wywołanie głównej funkcji
if __name__ == "__main__":
    analiza_fuzzy_clustering_planet()
    print("\n=== INSTRUKCJA DLA GOOGLE COLAB ===")
    print("Aby uruchomić ten skrypt w Google Colab, zainstaluj bibliotekę scikit-fuzzy:")
    print("!pip install scikit-fuzzy yellowbrick")
    print("Następnie załaduj plik planets.csv do Colab i uruchom funkcję analiza_fuzzy_clustering_planet('planets.csv')") 