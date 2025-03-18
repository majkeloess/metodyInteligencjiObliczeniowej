from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ładowanie zbioru danych Iris
iris = load_iris()
X, y = iris.data, iris.target

# Przygotowanie perceptronu
random_state = 42
perceptron = Perceptron(max_iter=1000, random_state=random_state)

# =============== METODA 1: Różne proporcje podziału ===============
test_sizes = [0.1, 0.2, 0.3]
results_prop = []

print("METODA 1: Różne proporcje podziału")
print("-" * 50)

for test_size in test_sizes:
    # Podział danych
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standaryzacja cech
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Uczenie modelu
    perceptron.fit(X_train_scaled, y_train)
    
    # Ocena modelu
    y_pred = perceptron.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results_prop.append({
        'test_size': test_size,
        'train_size': 1 - test_size,
        'accuracy': acc,
        'confusion_matrix': cm
    })
    
    print(f"\nTest size: {test_size} (Train size: {1-test_size})")
    print(f"Dokładność: {acc:.2%}")
    print("Macierz pomyłek:")
    print(cm)

# =============== METODA 2: Z stratyfikacją i bez stratyfikacji ===============
strat_options = [True, False]
results_strat = []

print("\n\nMETODA 2: Podział z stratyfikacją i bez stratyfikacji")
print("-" * 50)

for stratify_option in strat_options:
    # Podział danych
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, 
        stratify=y if stratify_option else None
    )
    
    # Standaryzacja cech
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Uczenie modelu
    perceptron.fit(X_train_scaled, y_train)
    
    # Ocena modelu
    y_pred = perceptron.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Sprawdzenie rozkładu klas
    train_class_dist = pd.Series(y_train).value_counts(normalize=True)
    test_class_dist = pd.Series(y_test).value_counts(normalize=True)
    
    results_strat.append({
        'stratify': stratify_option,
        'accuracy': acc,
        'confusion_matrix': cm,
        'train_dist': train_class_dist,
        'test_dist': test_class_dist
    })
    
    print(f"\nStratyfikacja: {stratify_option}")
    print(f"Dokładność: {acc:.2%}")
    print("Rozkład klas w zbiorze treningowym:")
    for cls, pct in train_class_dist.items():
        print(f"  Klasa {iris.target_names[cls]}: {pct:.2%}")
    print("Rozkład klas w zbiorze testowym:")
    for cls, pct in test_class_dist.items():
        print(f"  Klasa {iris.target_names[cls]}: {pct:.2%}")
    print("Macierz pomyłek:")
    print(cm)

# =============== METODA 3: Walidacja krzyżowa k-fold ===============
k_values = [3, 5, 10]
results_kfold = []
results_stratified_kfold = []

print("\n\nMETODA 3: Walidacja krzyżowa (k-fold)")
print("-" * 50)

# Standaryzacja całego zbioru danych - ważne dla walidacji krzyżowej
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

for k in k_values:
    # Zwykły k-fold
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_accuracies = []
    
    print(f"\nZwykły {k}-fold cross-validation:")
    
    for i, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        # Podział danych
        X_fold_train, X_fold_test = X_scaled[train_idx], X_scaled[test_idx]
        y_fold_train, y_fold_test = y[train_idx], y[test_idx]
        
        # Uczenie modelu
        perceptron.fit(X_fold_train, y_fold_train)
        
        # Ocena modelu
        y_fold_pred = perceptron.predict(X_fold_test)
        fold_acc = accuracy_score(y_fold_test, y_fold_pred)
        fold_accuracies.append(fold_acc)
        
        print(f"  Fold {i+1}: Dokładność = {fold_acc:.2%}")
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    results_kfold.append({
        'k': k,
        'accuracies': fold_accuracies,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc
    })
    
    print(f"  Średnia dokładność: {mean_acc:.2%}")
    print(f"  Odchylenie standardowe: {std_acc:.4f}")
    
    # Stratyfikowany k-fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    strat_fold_accuracies = []
    
    print(f"\nStratyfikowany {k}-fold cross-validation:")
    
    for i, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        # Podział danych
        X_fold_train, X_fold_test = X_scaled[train_idx], X_scaled[test_idx]
        y_fold_train, y_fold_test = y[train_idx], y[test_idx]
        
        # Uczenie modelu
        perceptron.fit(X_fold_train, y_fold_train)
        
        # Ocena modelu
        y_fold_pred = perceptron.predict(X_fold_test)
        fold_acc = accuracy_score(y_fold_test, y_fold_pred)
        strat_fold_accuracies.append(fold_acc)
        
        print(f"  Fold {i+1}: Dokładność = {fold_acc:.2%}")
    
    strat_mean_acc = np.mean(strat_fold_accuracies)
    strat_std_acc = np.std(strat_fold_accuracies)
    
    results_stratified_kfold.append({
        'k': k,
        'accuracies': strat_fold_accuracies,
        'mean_accuracy': strat_mean_acc,
        'std_accuracy': strat_std_acc
    })
    
    print(f"  Średnia dokładność: {strat_mean_acc:.2%}")
    print(f"  Odchylenie standardowe: {strat_std_acc:.4f}")

# =============== Podsumowanie wyników ===============
print("\n\nPODSUMOWANIE WYNIKÓW")
print("="*50)

# Podsumowanie dla różnych proporcji
print("\nWpływ proporcji podziału:")
for r in results_prop:
    print(f"  Test size: {r['test_size']}, Dokładność: {r['accuracy']:.2%}")

# Podsumowanie dla stratyfikacji
print("\nWpływ stratyfikacji:")
for r in results_strat:
    print(f"  Stratyfikacja: {r['stratify']}, Dokładność: {r['accuracy']:.2%}")

# Podsumowanie dla k-fold
print("\nWpływ walidacji krzyżowej (zwykły k-fold):")
for r in results_kfold:
    print(f"  k={r['k']}, Średnia dokładność: {r['mean_accuracy']:.2%}, Odchylenie: {r['std_accuracy']:.4f}")

# Podsumowanie dla stratyfikowanego k-fold
print("\nWpływ walidacji krzyżowej (stratyfikowany k-fold):")
for r in results_stratified_kfold:
    print(f"  k={r['k']}, Średnia dokładność: {r['mean_accuracy']:.2%}, Odchylenie: {r['std_accuracy']:.4f}")

# Wizualizacja porównawcza
plt.figure(figsize=(15, 10))

# Wykres dla różnych proporcji
plt.subplot(2, 2, 1)
test_sizes = [r['test_size'] for r in results_prop]
accuracies = [r['accuracy'] for r in results_prop]
plt.bar(range(len(test_sizes)), accuracies, color='skyblue')
plt.xticks(range(len(test_sizes)), [f"{ts:.1f}" for ts in test_sizes])
plt.xlabel('Rozmiar zbioru testowego')
plt.ylabel('Dokładność')
plt.title('Wpływ proporcji podziału na dokładność')
plt.ylim(0.8, 1.0)

# Wykres dla stratyfikacji
plt.subplot(2, 2, 2)
strat_options = ["Tak" if r['stratify'] else "Nie" for r in results_strat]
accuracies = [r['accuracy'] for r in results_strat]
plt.bar(range(len(strat_options)), accuracies, color='lightgreen')
plt.xticks(range(len(strat_options)), strat_options)
plt.xlabel('Zastosowanie stratyfikacji')
plt.ylabel('Dokładność')
plt.title('Wpływ stratyfikacji na dokładność')
plt.ylim(0.8, 1.0)

# Wykres dla k-fold
plt.subplot(2, 2, 3)
k_values = [r['k'] for r in results_kfold]
mean_accs = [r['mean_accuracy'] for r in results_kfold]
std_accs = [r['std_accuracy'] for r in results_kfold]

plt.bar(range(len(k_values)), mean_accs, yerr=std_accs, color='salmon', capsize=5)
plt.xticks(range(len(k_values)), [f"k={k}" for k in k_values])
plt.xlabel('Liczba foldów')
plt.ylabel('Średnia dokładność')
plt.title('Wyniki zwykłej walidacji krzyżowej')
plt.ylim(0.8, 1.0)

# Wykres dla stratyfikowanego k-fold
plt.subplot(2, 2, 4)
k_values = [r['k'] for r in results_stratified_kfold]
mean_accs = [r['mean_accuracy'] for r in results_stratified_kfold]
std_accs = [r['std_accuracy'] for r in results_stratified_kfold]

plt.bar(range(len(k_values)), mean_accs, yerr=std_accs, color='lightpurple', capsize=5)
plt.xticks(range(len(k_values)), [f"k={k}" for k in k_values])
plt.xlabel('Liczba foldów')
plt.ylabel('Średnia dokładność')
plt.title('Wyniki stratyfikowanej walidacji krzyżowej')
plt.ylim(0.8, 1.0)

plt.tight_layout()
plt.savefig('split_comparison_results.png')

# Wnioski
print("\nWNIOSKI:")
print("-"*50)
print("1. Wpływ proporcji podziału: Zmiana proporcji zbioru treningowego do testowego")
print("   wpływa na wyniki modelu, ponieważ mniejszy zbiór treningowy poskutkuje mniejszą ilością")
print("   danych do uczenia, a mniejszy zbiór testowy może prowadzić do bardziej zmiennych wyników.")
print()
print("2. Stratyfikacja: Zapewnienie podobnego rozkładu klas w zbiorach treningowym i testowym")
print("   jest szczególnie ważne dla niezbalansowanych zbiorów danych lub małych zbiorów, jak Iris.")
print("   Brak stratyfikacji może prowadzić do niedoreprezentowania niektórych klas w zbiorze testowym.")
print()
print("3. Walidacja krzyżowa: Zapewnia bardziej wiarygodną ocenę modelu, ponieważ każda próbka")
print("   jest używana zarówno do treningu jak i do testowania. Stratyfikowany k-fold dodatkowo")
print("   zapewnia zbalansowany rozkład klas w każdym foldu.")
print()
print("4. Wariancja wyników: Różne metody podziału danych mogą prowadzić do różnych wyników,")
print("   co pokazuje, jak ważne jest stosowanie rzetelnych metod walidacji, takich jak")
print("   stratyfikowana walidacja krzyżowa, aby uzyskać wiarygodną ocenę modelu.") 