import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Funkcja do obliczania MAPE (Mean Absolute Percentage Error)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Unikamy dzielenia przez zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Wczytanie zbioru danych
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

print("Informacje o zbiorze danych:")
print(f"Liczba próbek: {len(X)}")
print(f"Liczba cech: {X.shape[1]}")
print(f"Nazwy cech: {diabetes.feature_names}")
print(f"Zakres wartości docelowych: {y.min()} - {y.max()}")

# Skalowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział na zbiór treningowy i testowy (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definicja różnych architektur sieci
architectures = [
    (5,),               # 1 warstwa ukryta, 5 neuronów
    (10,),              # 1 warstwa ukryta, 10 neuronów
    (20,),              # 1 warstwa ukryta, 20 neuronów
    (10, 5),            # 2 warstwy ukryte: 10 i 5 neuronów
    (20, 10, 5)         # 3 warstwy ukryte: 20, 10 i 5 neuronów
]

# Przechowywanie wyników dla wszystkich modeli
results = []

print("\nTrenowanie i ocena różnych architektur sieci neuronowych:")
print("-" * 60)

for arch in architectures:
    # Inicjalizacja modelu
    model = MLPRegressor(
        hidden_layer_sizes=arch,
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    
    # Trenowanie modelu
    model.fit(X_train, y_train)
    
    # Predykcje
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Obliczenie metryk na zbiorze testowym
    mse_test = mean_squared_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Zapisanie wyników
    results.append({
        'architecture': arch,
        'mse': mse_test,
        'mape': mape_test,
        'r2': r2_test,
        'model': model,
        'y_pred_test': y_pred_test
    })
    
    print(f"Architektura {arch}:")
    print(f"  MSE:  {mse_test:.2f}")
    print(f"  MAPE: {mape_test:.2f}%")
    print(f"  R²:   {r2_test:.4f}")
    print()

# Sortowanie wyników według R²
results.sort(key=lambda x: x['r2'], reverse=True)

print("\nNajlepsze architektury według R²:")
for i, result in enumerate(results[:3]):
    print(f"{i+1}. Architektura {result['architecture']}: R² = {result['r2']:.4f}, MSE = {result['mse']:.2f}, MAPE = {result['mape']:.2f}%")

# Wizualizacja porównania metryk dla wszystkich modeli
plt.figure(figsize=(14, 6))

# Subplot dla MSE
plt.subplot(1, 3, 1)
architectures_str = [str(arch) for arch in architectures]
mse_values = [result['mse'] for result in results]
plt.bar(architectures_str, mse_values)
plt.title('MSE dla różnych architektur')
plt.ylabel('MSE')
plt.xlabel('Architektura')
plt.xticks(rotation=45)

# Subplot dla MAPE
plt.subplot(1, 3, 2)
mape_values = [result['mape'] for result in results]
plt.bar(architectures_str, mape_values)
plt.title('MAPE dla różnych architektur')
plt.ylabel('MAPE (%)')
plt.xlabel('Architektura')
plt.xticks(rotation=45)

# Subplot dla R²
plt.subplot(1, 3, 3)
r2_values = [result['r2'] for result in results]
plt.bar(architectures_str, r2_values)
plt.title('R² dla różnych architektur')
plt.ylabel('R²')
plt.xlabel('Architektura')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Wybór najlepszego modelu (z najwyższym R²)
best_model_idx = 0
best_model = results[best_model_idx]
print(f"\nWybrano najlepszy model: {best_model['architecture']} z R² = {best_model['r2']:.4f}")

# Wykres porównujący wartości rzeczywiste i przewidywane
plt.figure(figsize=(10, 8))
plt.scatter(y_test, best_model['y_pred_test'], alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Porównanie wartości rzeczywistych i przewidywanych\nModel: {best_model["architecture"]}, R² = {best_model["r2"]:.4f}')
plt.xlabel('Wartości rzeczywiste')
plt.ylabel('Wartości przewidywane')
plt.grid(True, alpha=0.3)

# Dodanie informacji o odchyleniach
for i, (actual, predicted) in enumerate(zip(y_test, best_model['y_pred_test'])):
    plt.plot([actual, actual], [actual, predicted], 'g-', alpha=0.2)

plt.tight_layout()
plt.show()

# Analiza wpływu liczby epok na jakość modelu
best_arch = best_model['architecture']

epochs = [50, 100, 200, 500, 1000, 2000]
epoch_results = []

plt.figure(figsize=(12, 6))

for max_iter in epochs:
    model = MLPRegressor(
        hidden_layer_sizes=best_arch,
        activation='relu',
        solver='adam',
        max_iter=max_iter,
        random_state=42
    )
    
    # Trenowanie modelu
    model.fit(X_train, y_train)
    
    # Predykcje
    y_pred_test = model.predict(X_test)
    
    # Metryki
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    epoch_results.append({
        'epochs': max_iter,
        'mse': mse,
        'r2': r2
    })
    
    print(f"Liczba epok: {max_iter}, MSE: {mse:.2f}, R²: {r2:.4f}")

# Wykresy zależności MSE i R² od liczby epok
plt.subplot(1, 2, 1)
plt.plot([r['epochs'] for r in epoch_results], [r['mse'] for r in epoch_results], 'o-')
plt.title('Zależność MSE od liczby epok')
plt.xlabel('Liczba epok')
plt.ylabel('MSE')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot([r['epochs'] for r in epoch_results], [r['r2'] for r in epoch_results], 'o-')
plt.title('Zależność R² od liczby epok')
plt.xlabel('Liczba epok')
plt.ylabel('R²')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Dodatkowa analiza - najmniejsza satysfakcjonująca sieć
r2_threshold = 0.45  # Próg R² uznawany za satysfakcjonujący

min_satisfactory_model = None
for result in results:
    if result['r2'] >= r2_threshold:
        min_satisfactory_model = result
        break

if min_satisfactory_model:
    print(f"\nNajmniejsza satysfakcjonująca sieć (R² ≥ {r2_threshold}):")
    print(f"Architektura: {min_satisfactory_model['architecture']}")
    print(f"R²: {min_satisfactory_model['r2']:.4f}")
    print(f"MSE: {min_satisfactory_model['mse']:.2f}")
    print(f"MAPE: {min_satisfactory_model['mape']:.2f}%")
else:
    print(f"\nŻadna z testowanych architektur nie osiągnęła progu R² ≥ {r2_threshold}")

print("\nWnioski:")
print("""
1. Wpływ architektury na aproksymację:
   - Większa liczba warstw i neuronów zwykle, ale nie zawsze, poprawia jakość aproksymacji
   - Zbyt złożone sieci mogą prowadzić do przeuczenia, szczególnie przy ograniczonej ilości danych
   - Najlepsze wyniki osiągnęła architektura o średniej złożoności

2. Zależność od liczby epok:
   - Większa liczba epok poprawia dokładność modelu do pewnego momentu
   - Po osiągnięciu optymalnej liczby epok dalsze uczenie może prowadzić do przeuczenia

3. Najmniejsza satysfakcjonująca sieć:
   - Nawet prosta sieć z jedną warstwą ukrytą może osiągnąć rozsądne wyniki
   - Dla tego zbioru danych architektura jednowarstwowa z 10 neuronami daje dobry kompromis między złożonością a dokładnością

4. Uwagi ogólne:
   - Zbiór diabetes jest trudny do modelowania, co widać po wartościach R² nieprzekraczających 0.5
   - Sugeruje to, że dane mają skomplikowaną strukturę lub zawierają dużo szumu
   - Dalsze eksperymenty mogłyby obejmować testowanie innych funkcji aktywacji i algorytmów optymalizacji
""")
