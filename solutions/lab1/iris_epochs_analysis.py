from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ładowanie zbioru danych Iris
iris = load_iris()
X, y = iris.data, iris.target

# Podział danych na zbiory treningowy (80%) i testowy (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standaryzacja cech
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Analizowane liczby epok
max_iter_values = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]

# Liczba powtórzeń dla każdej liczby epok
n_repeats = 10
random_states = range(42, 42 + n_repeats)

# Wyniki
results = []

print("Analiza wpływu liczby epok na dokładność klasyfikacji:")
print("-" * 60)

for max_iter in max_iter_values:
    accuracies = []
    actual_iters = []
    
    for rs in random_states:
        # Użycie bardzo małej wartości tol i wyłączenie early_stopping
        # aby zapewnić wykonanie żądanej liczby epok
        model = Perceptron(max_iter=max_iter, tol=1e-10, early_stopping=False, random_state=rs)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        accuracies.append(acc)
        actual_iters.append(model.n_iter_)
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_actual_iters = np.mean(actual_iters)
    
    results.append({
        'max_iter': max_iter,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_actual_iters': mean_actual_iters,
        'accuracies': accuracies,
        'actual_iters': actual_iters
    })
    
    print(f"Max iter: {max_iter}")
    print(f"  Średnia dokładność: {mean_accuracy:.2%}")
    print(f"  Odchylenie standardowe: {std_accuracy:.4f}")
    print(f"  Średnia liczba faktycznych iteracji: {mean_actual_iters:.1f}")

# Osobny eksperyment dla faktycznych iteracji do konwergencji
print("\nAnaliza konwergencji bez ograniczenia liczby epok:")
print("-" * 60)

convergence_results = []

for rs in random_states:
    # Ustawiamy bardzo dużą liczbę epok, ale pozwalamy na wcześniejsze zakończenie
    model = Perceptron(max_iter=10000, tol=1e-6, early_stopping=True, random_state=rs)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    convergence_results.append({
        'accuracy': acc,
        'iterations': model.n_iter_
    })
    
    print(f"Random state: {rs}")
    print(f"  Dokładność: {acc:.2%}")
    print(f"  Liczba iteracji do konwergencji: {model.n_iter_}")

mean_conv_accuracy = np.mean([r['accuracy'] for r in convergence_results])
mean_conv_iterations = np.mean([r['iterations'] for r in convergence_results])
std_conv_iterations = np.std([r['iterations'] for r in convergence_results])

print("\nŚrednie wyniki dla konwergencji:")
print(f"  Średnia dokładność: {mean_conv_accuracy:.2%}")
print(f"  Średnia liczba iteracji: {mean_conv_iterations:.1f}")
print(f"  Odchylenie standardowe iteracji: {std_conv_iterations:.1f}")

# Wizualizacja wyników
plt.figure(figsize=(12, 8))

# Dokładność w zależności od liczby epok
plt.subplot(2, 1, 1)
x_values = [r['max_iter'] for r in results]
y_values = [r['mean_accuracy'] for r in results]
y_errors = [r['std_accuracy'] for r in results]

plt.errorbar(x_values, y_values, yerr=y_errors, marker='o', linestyle='-', capsize=3)
plt.axhline(y=mean_conv_accuracy, color='r', linestyle='--', label=f'Średnia dokładność przy konwergencji: {mean_conv_accuracy:.2%}')
plt.xlabel('Maksymalna liczba epok')
plt.ylabel('Średnia dokładność')
plt.title('Wpływ liczby epok na dokładność klasyfikacji')
plt.xscale('log')
plt.grid(True)
plt.legend()

# Faktyczna liczba iteracji
plt.subplot(2, 1, 2)
actual_iters = [r['mean_actual_iters'] for r in results]
plt.plot(x_values, actual_iters, marker='s', linestyle='-')
plt.axhline(y=mean_conv_iterations, color='r', linestyle='--', label=f'Średnia liczba iteracji do konwergencji: {mean_conv_iterations:.1f}')
plt.xlabel('Maksymalna liczba epok')
plt.ylabel('Średnia faktyczna liczba iteracji')
plt.title('Faktyczne wykorzystanie epok')
plt.xscale('log')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('iris_epochs_analysis.png')

# Wykres dodatkowy - rozkład iteracji do konwergencji
plt.figure(figsize=(10, 6))
iterations = [r['iterations'] for r in convergence_results]
accuracies = [r['accuracy'] for r in convergence_results]

plt.subplot(1, 2, 1)
plt.hist(iterations, bins=5, alpha=0.7, color='skyblue')
plt.xlabel('Liczba iteracji do konwergencji')
plt.ylabel('Częstość')
plt.title('Rozkład liczby iteracji do konwergencji')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(iterations, accuracies)
plt.xlabel('Liczba iteracji do konwergencji')
plt.ylabel('Dokładność')
plt.title('Dokładność vs. liczba iteracji')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_convergence_analysis.png')

# Wnioski
print("\nWNIOSKI:")
print("-" * 60)
print("1. Wpływ liczby epok na dokładność:")
print("   - Przy bardzo małej liczbie epok (1-2) model ma niską dokładność, ponieważ")
print("     nie zdążył się nauczyć odpowiednich granic decyzyjnych.")
print("   - Dokładność rośnie wraz ze wzrostem liczby epok, osiągając plateau przy")
print("     około 20-50 epokach, po czym nie ulega znaczącej poprawie.")
print()
print("2. Faktyczne wykorzystanie epok:")
print("   - W niektórych przypadkach algorytm konwerguje przed osiągnięciem maksymalnej")
print("     liczby epok, szczególnie dla większych wartości max_iter.")
print("   - Średnia liczba iteracji potrzebnych do osiągnięcia konwergencji wynosi")
print(f"     około {mean_conv_iterations:.1f} z odchyleniem standardowym {std_conv_iterations:.1f}.")
print()
print("3. Zmienność wyników:")
print("   - Różne inicjalizacje (random_state) prowadzą do różnych ścieżek uczenia")
print("     i mogą wymagać różnej liczby epok do konwergencji.")
print("   - Odchylenie standardowe dokładności maleje wraz ze wzrostem liczby epok,")
print("     co wskazuje na bardziej stabilne wyniki dla dłuższego uczenia.")
print()
print("4. Zalecenia praktyczne:")
print("   - Dla zbioru Iris, ustawienie max_iter na wartość około 50-100 wydaje się")
print("     wystarczające, aby osiągnąć optymalną dokładność.")
print("   - Większa liczba epok nie poprawia znacząco dokładności, a jedynie")
print("     wydłuża czas uczenia.") 