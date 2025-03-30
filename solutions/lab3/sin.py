import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Generowanie danych treningowych
X_range = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1)
y_true = np.sin(X_range).ravel()

# Testowanie różnych rozmiarów warstwy ukrytej, aby znaleźć najmniejszą skuteczną sieć
hidden_sizes = [1, 2, 3, 4, 5, 10]
models = []
mse_scores = []

for size in hidden_sizes:
    # Inicjalizacja modelu z funkcją aktywacji tanh
    model = MLPRegressor(
        hidden_layer_sizes=(size,),  # jedna warstwa ukryta
        activation='tanh',
        solver='adam',
        max_iter=2000,
        random_state=42
    )
    
    # Trenowanie modelu
    model.fit(X_range, y_true)
    
    # Predykcja
    y_pred = model.predict(X_range)
    
    # Obliczenie błędu MSE
    mse = mean_squared_error(y_true, y_pred)
    
    models.append(model)
    mse_scores.append(mse)
    
    print(f"Liczba neuronów: {size}, MSE: {mse:.6f}")

# Znalezienie najlepszego modelu (z najmniejszą liczbą neuronów poniżej progu błędu)
threshold = 0.001  # próg akceptowalnego błędu
best_index = 0
for i, mse in enumerate(mse_scores):
    if mse < threshold:
        best_index = i
        break

best_model = models[best_index]
best_size = hidden_sizes[best_index]
print(f"\nNajlepsza wybrana sieć ma {best_size} neuronów w warstwie ukrytej, MSE: {mse_scores[best_index]:.6f}")

# Dostęp do wag i biasów
weights_input_hidden = best_model.coefs_[0]  # wagi między warstwą wejściową a ukrytą
weights_hidden_output = best_model.coefs_[1]  # wagi między warstwą ukrytą a wyjściową
bias_hidden = best_model.intercepts_[0]  # biasy warstwy ukrytej
bias_output = best_model.intercepts_[1]  # bias warstwy wyjściowej

# Przygotowanie danych do wizualizacji
X_test = np.linspace(-2*np.pi, 2*np.pi, 500).reshape(-1, 1)
y_true_test = np.sin(X_test).ravel()
y_pred_test = best_model.predict(X_test)

# Wizualizacja wyników
plt.figure(figsize=(10, 6))
plt.plot(X_test, y_true_test, 'b-', label='$f(x) = \sin(x)$')
plt.plot(X_test, y_pred_test, 'r--', label=f'Aproksymacja (MLP, {best_size} neurony)')
plt.legend()
plt.grid(True)
plt.title(f'Aproksymacja funkcji $\sin(x)$ przy użyciu sieci neuronowej z {best_size} neuronami')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('sin_approximation.png')
plt.show()

# Generowanie wzoru funkcji aproksymującej
formula = f"f(x) = "
for i in range(best_size):
    w1 = weights_input_hidden[0, i]
    b1 = bias_hidden[i]
    w2 = weights_hidden_output[i, 0]
    
    if i > 0:
        if w2 > 0:
            formula += " + "
        else:
            formula += " - "
        formula += f"{abs(w2):.4f} * tanh({w1:.4f} * x + {b1:.4f})"
    else:
        formula += f"{w2:.4f} * tanh({w1:.4f} * x + {b1:.4f})"

formula += f" + {bias_output[0]:.4f}"

print("\nWzór funkcji aproksymującej:")
print(formula)

# Analiza dokładności aproksymacji w różnych przedziałach
intervals = [
    (-2*np.pi, -np.pi),
    (-np.pi, 0),
    (0, np.pi),
    (np.pi, 2*np.pi)
]

for a, b in intervals:
    # Tworzenie maski dla wartości X w danym przedziale
    mask = (X_test.ravel() >= a) & (X_test.ravel() <= b)
    
    # Wybieranie odpowiednich wartości przy użyciu maski
    y_true_interval = y_true_test[mask]
    y_pred_interval = y_pred_test[mask]
    
    # Obliczenie MSE dla przedziału
    interval_mse = mean_squared_error(y_true_interval, y_pred_interval)
    print(f"MSE w przedziale [{a:.2f}, {b:.2f}]: {interval_mse:.6f}")