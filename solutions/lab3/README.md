# Aproksymacja funkcji zysków za pomocą sieci neuronowej

## Opis zadania

Projekt polega na wykorzystaniu sieci neuronowej do aproksymacji funkcji zysków ze sprzedaży produktu na podstawie wydatków na reklamę w różnych mediach. Funkcja zysków Z(w_TV, w_radio, w_prasa) zależy od trzech zmiennych wejściowych:
- w_TV - wydatki na reklamę telewizyjną
- w_radio - wydatki na reklamę radiową
- w_prasa - wydatki na reklamę prasową

## Dane

Dane pochodzą z pliku `Advertising.csv` i zawierają informacje o wydatkach na reklamę w trzech mediach oraz odpowiadające im zyski ze sprzedaży.

## Rozwiązanie

W projekcie porównane zostały różne architektury sieci neuronowej, różniące się między sobą:
- liczbą neuronów w warstwach ukrytych (8 i 16)
- funkcją aktywacji (ReLU i tanh)

Dla każdej kombinacji tych parametrów obliczono błąd średniokwadratowy (MSE) na zbiorze testowym, co pozwoliło wybrać najlepszy model.

## Wymagania

Do uruchomienia programu potrzebne są następujące biblioteki:
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- TensorFlow
- Seaborn

Można je zainstalować za pomocą:

```
pip install numpy pandas matplotlib scikit-learn tensorflow seaborn
```

## Uruchomienie

Aby uruchomić program, należy wykonać:

```
python neural_network_aprox.py
```

## Wyniki

Po uruchomieniu programu zostaną wygenerowane:
1. Wykresy prezentujące historię treningu najlepszego modelu
2. Porównanie rzeczywistych i przewidywanych wartości
3. Wykres porównujący MSE dla różnych architektur sieci
4. Wykres przedstawiający ważność poszczególnych cech
5. Mapa ciepła pokazująca korelacje między zmiennymi
6. Raport w pliku `raport.txt` zawierający szczegółowe wyniki i wnioski

## Struktura folderów

- `neural_network_aprox.py` - główny skrypt zawierający implementację
- `raport.txt` - raport z wynikami eksperymentów (generowany po uruchomieniu)
- Pliki graficzne:
  - `best_model_results.png` - wykresy dla najlepszego modelu
  - `models_comparison.png` - porównanie MSE dla różnych architektur
  - `feature_importance.png` - ważność poszczególnych cech
  - `correlation_heatmap.png` - mapa korelacji między zmiennymi 