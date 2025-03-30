import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns

# Wczytanie danych
data = pd.read_csv('../data/Advertising.csv')

# Sprawdzenie danych
print("Pierwsze 5 wierszy danych:")
print(data.head())
print("\nInformacje o danych:")
print(data.info())
print("\nStatystyki opisowe:")
print(data.describe())

# Przygotowanie danych
X = data.iloc[:, 1:4].values  # TV, Radio, Newspaper
y = data.iloc[:, 4].values    # Sales

# Podział na zbiór treningowy i testowy (80% - trening, 20% - test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standaryzacja danych
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Funkcja do tworzenia modelu
def create_model(neurons, activation):
    model = Sequential([
        Dense(neurons, activation=activation, input_shape=(3,)),
        Dense(neurons, activation=activation),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Testowanie różnych architektur
architectures = [
    {'neurons': 8, 'activation': 'relu'},
    {'neurons': 16, 'activation': 'relu'},
    {'neurons': 8, 'activation': 'tanh'},
    {'neurons': 16, 'activation': 'tanh'}
]

results = []

for arch in architectures:
    print(f"\nTrening sieci z {arch['neurons']} neuronami i funkcją aktywacji {arch['activation']}")
    
    # Utworzenie modelu
    model = create_model(arch['neurons'], arch['activation'])
    
    # Trening modelu
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=200,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    # Predykcja na zbiorze testowym
    y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
    
    # Przywrócenie oryginalnej skali
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Obliczenie błędu MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE dla modelu: {mse:.4f}")
    
    # Zapisanie wyników
    results.append({
        'architecture': f"{arch['neurons']} neurons, {arch['activation']}",
        'mse': mse,
        'model': model,
        'history': history
    })

# Sortowanie wyników według MSE
results.sort(key=lambda x: x['mse'])

# Wyświetlenie porównania wyników
print("\nPorównanie wyników MSE:")
for result in results:
    print(f"{result['architecture']}: {result['mse']:.4f}")

# Wizualizacja wyników najlepszego modelu
best_result = results[0]
best_model = best_result['model']
print(f"\nNajlepszy model: {best_result['architecture']} z MSE {best_result['mse']:.4f}")

# Wykres historii treningu najlepszego modelu
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(best_result['history'].history['loss'], label='Trening')
plt.plot(best_result['history'].history['val_loss'], label='Walidacja')
plt.title(f'Historia treningu - {best_result["architecture"]}')
plt.xlabel('Epoka')
plt.ylabel('Błąd MSE')
plt.legend()

# Wykres porównania rzeczywistych i przewidywanych wartości
plt.subplot(1, 2, 2)
y_pred_best = best_model.predict(X_test_scaled, verbose=0).flatten()
y_pred_best = scaler_y.inverse_transform(y_pred_best.reshape(-1, 1)).flatten()

plt.scatter(y_test, y_pred_best, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Porównanie rzeczywistych i przewidywanych wartości')
plt.xlabel('Rzeczywiste wartości')
plt.ylabel('Przewidywane wartości')

plt.tight_layout()
plt.savefig('lab3/best_model_results.png')
plt.show()

# Wizualizacja porównania wszystkich modeli
plt.figure(figsize=(10, 6))
architectures_labels = [result['architecture'] for result in results]
mse_values = [result['mse'] for result in results]

bars = plt.bar(architectures_labels, mse_values)
plt.title('Porównanie MSE dla różnych architektur')
plt.ylabel('MSE')
plt.xlabel('Architektura sieci')
plt.xticks(rotation=45)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('lab3/models_comparison.png')
plt.show()

# Interpretacja wyników
print("\nAnaliza wpływu parametrów na wynik:")
for i in range(len(results)-1):
    for j in range(i+1, len(results)):
        arch1 = results[i]
        arch2 = results[j]
        
        # Porównywanie modeli z tą samą funkcją aktywacji, ale różną liczbą neuronów
        if arch1['architecture'].split(', ')[1] == arch2['architecture'].split(', ')[1]:
            print(f"Porównanie liczby neuronów przy funkcji {arch1['architecture'].split(', ')[1]}:")
            print(f"  {arch1['architecture']}: MSE = {arch1['mse']:.4f}")
            print(f"  {arch2['architecture']}: MSE = {arch2['mse']:.4f}")
            print(f"  Różnica MSE: {abs(arch1['mse'] - arch2['mse']):.4f}")
            
        # Porównywanie modeli z tą samą liczbą neuronów, ale różną funkcją aktywacji
        if arch1['architecture'].split(' ')[0] == arch2['architecture'].split(' ')[0]:
            print(f"Porównanie funkcji aktywacji przy {arch1['architecture'].split(' ')[0]} neuronach:")
            print(f"  {arch1['architecture']}: MSE = {arch1['mse']:.4f}")
            print(f"  {arch2['architecture']}: MSE = {arch2['mse']:.4f}")
            print(f"  Różnica MSE: {abs(arch1['mse'] - arch2['mse']):.4f}")

# Analiza ważności cech
# Tworzymy prosty model liniowy do oceny ważności cech
linear_model = Sequential([Dense(1, activation='linear', input_shape=(3,))])
linear_model.compile(optimizer='adam', loss='mse')
linear_model.fit(X_train_scaled, y_train_scaled, epochs=100, verbose=0)

weights = linear_model.get_weights()[0].flatten()
feature_names = ['TV', 'Radio', 'Newspaper']
features_importance = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(weights)})
features_importance = features_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 5))
plt.bar(features_importance['Feature'], features_importance['Importance'])
plt.title('Ważność cech w modelu liniowym')
plt.ylabel('Ważność (wartość bezwzględna wag)')
plt.savefig('lab3/feature_importance.png')
plt.show()

print("\nWażność cech:")
for index, row in features_importance.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")

# Wizualizacja zależności między zmiennymi
plt.figure(figsize=(12, 10))
sns.heatmap(data.iloc[:, 1:].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelacja między zmiennymi')
plt.savefig('lab3/correlation_heatmap.png')
plt.show()

# Zapisanie raportu
with open('lab3/raport.txt', 'w') as f:
    f.write("RAPORT Z EKSPERYMENTÓW Z SIECIĄ NEURONOWĄ\n")
    f.write("========================================\n\n")
    
    f.write("Opis problemu:\n")
    f.write("Zadanie polegało na aproksymacji funkcji zysków Z(w_TV, w_radio, w_prasa) za pomocą\n")
    f.write("sieci neuronowej, gdzie w_TV, w_radio, w_prasa to wydatki na reklamę odpowiednio\n")
    f.write("w telewizji, radiu i prasie.\n\n")
    
    f.write("Dane:\n")
    f.write(f"- Liczba próbek: {len(data)}\n")
    f.write(f"- Zmienne wejściowe: TV, Radio, Newspaper\n")
    f.write(f"- Zmienna wyjściowa: Sales (zyski)\n\n")
    
    f.write("Przygotowanie danych:\n")
    f.write("- Podział na zbiór treningowy (80%) i testowy (20%)\n")
    f.write("- Standaryzacja danych wejściowych i wyjściowych\n\n")
    
    f.write("Testowane architektury sieci:\n")
    for arch in architectures:
        f.write(f"- {arch['neurons']} neuronów w warstwach ukrytych, funkcja aktywacji: {arch['activation']}\n")
    f.write("\n")
    
    f.write("Wyniki (MSE na zbiorze testowym):\n")
    for result in results:
        f.write(f"- {result['architecture']}: {result['mse']:.4f}\n")
    f.write("\n")
    
    f.write(f"Najlepszy model: {best_result['architecture']} z MSE {best_result['mse']:.4f}\n\n")
    
    f.write("Analiza ważności cech:\n")
    for index, row in features_importance.iterrows():
        f.write(f"- {row['Feature']}: {row['Importance']:.4f}\n")
    
    f.write("\nWnioski:\n")
    if 'relu' in best_result['architecture']:
        activation_conclusion = "ReLU okazała się lepszą funkcją aktywacji dla tego problemu."
    else:
        activation_conclusion = "Tanh okazała się lepszą funkcją aktywacji dla tego problemu."
    
    if '16' in best_result['architecture']:
        neurons_conclusion = "Większa liczba neuronów (16) dała lepsze wyniki."
    else:
        neurons_conclusion = "Mniejsza liczba neuronów (8) okazała się wystarczająca."
    
    f.write(f"1. {activation_conclusion}\n")
    f.write(f"2. {neurons_conclusion}\n")
    f.write("3. Ważność cech pokazuje, które zmienne mają największy wpływ na wynik.\n")
    f.write("4. Standaryzacja danych była kluczowa dla uzyskania dobrych wyników.\n")

print("\nUtworzono raport w pliku lab3/raport.txt") 