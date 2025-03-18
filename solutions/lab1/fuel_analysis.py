import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

# Upewnienie się, że wykresy zostaną zapisane w odpowiednim katalogu
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Wczytanie danych
data = pd.read_csv('../../data/fuel.txt')

# Konwersja klas A i B na 0 i 1
data['purity_class'] = data['purity_class'].map({'A': 0, 'B': 1})

# Podział na cechy i etykiety
X = data[['c_1', 'c_2', 'c_3']].values
y = data['purity_class'].values

# Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Funkcja do trenowania i oceny modelu
def train_and_evaluate(random_state):
    # Inicjalizacja perceptronu (pojedynczy neuron)
    perceptron = Perceptron(
        random_state=random_state,
        max_iter=1000,
        tol=1e-3
    )
    
    # Trenowanie modelu
    perceptron.fit(X_scaled, y)
    
    # Predykcja
    y_pred = perceptron.predict(X_scaled)
    
    # Obliczenie dokładności
    accuracy = accuracy_score(y, y_pred)
    
    return perceptron, accuracy

# Przeprowadzenie 5-krotnego uczenia
results = []
for i in range(5):
    model, acc = train_and_evaluate(random_state=i)
    results.append((i, model, acc))
    print(f"Przebieg {i+1}: Dokładność = {acc:.4f}")

# Znalezienie najlepszego modelu
best_run = max(results, key=lambda x: x[2])
print(f"\nNajlepszy wynik: Przebieg {best_run[0]+1} z dokładnością {best_run[2]:.4f}")

# Wykres dokładności dla każdego przebiegu
plt.figure(figsize=(10, 6))
plt.bar(range(1, 6), [r[2] for r in results])
plt.xlabel('Numer przebiegu')
plt.ylabel('Dokładność')
plt.title('Porównanie dokładności dla 5 przebiegów uczenia perceptronu')
plt.ylim(0, 1)
plt.xticks(range(1, 6))
plt.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(CURRENT_DIR, 'fuel_accuracy_comparison.png'))
plt.close()

# Wizualizacja danych w przestrzeni cech c_1 i c_3 (najważniejsze cechy)
plt.figure(figsize=(10, 6))
for class_val in [0, 1]:
    mask = y == class_val
    label = 'A' if class_val == 0 else 'B'
    plt.scatter(X[mask, 0], X[mask, 2], alpha=0.7, label=f'Klasa {label}')

# Dodanie granicy decyzyjnej dla najlepszego modelu
best_model = best_run[1]
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 2].min() - 1, X[:, 2].max() + 1, 100))
grid = np.c_[xx.ravel(), np.zeros_like(xx.ravel()), yy.ravel()]
grid_scaled = scaler.transform(grid)
Z = best_model.predict(grid_scaled).reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', linestyles=['-'], levels=[0.5])

plt.xlabel('c_1')
plt.ylabel('c_3')
plt.title('Rozkład próbek i granica decyzyjna w przestrzeni cech c_1 i c_3')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(CURRENT_DIR, 'fuel_data_visualization.png'))
plt.close()

# Analiza wag perceptronu
best_weights = best_run[1].coef_[0]
feature_names = ['c_1', 'c_2', 'c_3']
plt.figure(figsize=(8, 6))
plt.bar(feature_names, best_weights)
plt.xlabel('Cechy')
plt.ylabel('Wagi')
plt.title('Wagi perceptronu dla poszczególnych cech')
plt.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(CURRENT_DIR, 'fuel_weights.png'))
plt.close()

# Podsumowanie
print("\nPodsumowanie:")
print(f"Średnia dokładność: {np.mean([r[2] for r in results]):.4f}")
print(f"Odchylenie standardowe: {np.std([r[2] for r in results]):.4f}")
print(f"Min dokładność: {min([r[2] for r in results]):.4f}")
print(f"Max dokładność: {max([r[2] for r in results]):.4f}")
print("\nWagi najlepszego modelu:")
for name, weight in zip(feature_names, best_weights):
    print(f"{name}: {weight:.6f}")
print(f"Bias: {best_run[1].intercept_[0]:.6f}") 