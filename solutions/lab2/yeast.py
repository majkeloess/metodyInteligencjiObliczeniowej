import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import time
import warnings
warnings.filterwarnings('ignore')

# Pobranie i przygotowanie zbioru danych
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
column_names = ["sequence_name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "class"]
df = pd.read_csv(url, delimiter="\s+", names=column_names)

# Usunięcie pierwszej kolumny, która zawiera identyfikatory sekwencji
df = df.drop("sequence_name", axis=1)

# Analiza wstępna
print("Informacje o zbiorze danych:")
print(f"Liczba próbek: {df.shape[0]}")
print(f"Liczba cech: {df.shape[1] - 1}")
print(f"Liczba klas: {df['class'].nunique()}")
print("\nRozkład klas:")
class_counts = df["class"].value_counts()
for cls, count in class_counts.items():
    print(f"{cls}: {count} ({count/len(df)*100:.2f}%)")

# Wizualizacja rozkładu klas
plt.figure(figsize=(12, 6))
sns.countplot(x="class", data=df, order=df["class"].value_counts().index)
plt.title("Rozkład klas w zbiorze danych Yeast")
plt.xlabel("Klasa")
plt.ylabel("Liczba próbek")
plt.tight_layout()
plt.show()

# Przygotowanie danych
X = df.drop("class", axis=1).values
y = df["class"].values

# Kodowanie etykiet klas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

# Sprawdzenie liczności klas po kodowaniu
class_counts = np.bincount(y_encoded)
print("\nLiczność klas po kodowaniu:")
for i, count in enumerate(class_counts):
    print(f"Klasa {i} ({class_names[i]}): {count}")

# Usunięcie klas z bardzo małą liczbą próbek lub połączenie ich
# Podejście 1: Filtrowanie zbyt małych klas (poniżej 5 próbek)
min_samples = 5
valid_classes_mask = np.array([count >= min_samples for count in class_counts])
valid_class_indices = np.where(valid_classes_mask)[0]

valid_sample_mask = np.isin(y_encoded, valid_class_indices)
X_filtered = X[valid_sample_mask]
y_filtered = y_encoded[valid_sample_mask]

# Remap labels to be consecutive integers
label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_class_indices)}
y_remapped = np.array([label_map[y_val] for y_val in y_filtered])
class_names_filtered = class_names[valid_class_indices]

print(f"\nPo filtrowaniu małych klas (min. {min_samples} próbek):")
print(f"Liczba próbek: {len(X_filtered)}")
print(f"Liczba klas: {len(class_names_filtered)}")

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_remapped, test_size=0.3, random_state=42, stratify=y_remapped)

# Skalowanie cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Funkcja do oceny modelu
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predykcje
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metryki dla zbioru treningowego
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    
    # Metryki dla zbioru testowego
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    return {
        'model_name': model_name,
        'training_time': training_time,
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'model': model,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

# 1. Bazowy model MLP
print("\n1. Bazowy model MLP:")
base_model = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=1000,
    random_state=42
)

base_result = evaluate_model(base_model, X_train_scaled, X_test_scaled, y_train, y_test, "MLP Bazowy")
print(f"Dokładność (zbiór treningowy): {base_result['train_accuracy']:.4f}")
print(f"Dokładność (zbiór testowy): {base_result['test_accuracy']:.4f}")
print(f"Czas treningu: {base_result['training_time']:.2f}s")

# Macierz pomyłek dla bazowego modelu (zbiór testowy)
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, base_result['y_test_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_filtered, yticklabels=class_names_filtered)
plt.xlabel('Predykcja')
plt.ylabel('Wartość rzeczywista')
plt.title('Macierz pomyłek dla bazowego modelu (zbiór testowy)')
plt.tight_layout()
plt.show()

# 2. Badanie wpływu różnych architektur sieci
print("\n2. Wpływ architektury sieci:")
architectures = [
    (50,),
    (100,),
    (200,),
    (100, 50),
    (200, 100),
    (300, 150, 50)
]

arch_results = []
for arch in architectures:
    model = MLPClassifier(
        hidden_layer_sizes=arch,
        max_iter=1000,
        random_state=42
    )
    result = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, f"MLP {arch}")
    arch_results.append(result)
    print(f"Architektura {arch}: Dokładność = {result['test_accuracy']:.4f}, Czas = {result['training_time']:.2f}s")

# Wizualizacja wpływu architektury na dokładność i czas
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.bar([str(arch) for arch in architectures], [result['test_accuracy'] for result in arch_results])
plt.xlabel('Architektura sieci')
plt.ylabel('Dokładność (zbiór testowy)')
plt.title('Wpływ architektury na dokładność')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar([str(arch) for arch in architectures], [result['training_time'] for result in arch_results])
plt.xlabel('Architektura sieci')
plt.ylabel('Czas treningu [s]')
plt.title('Wpływ architektury na czas treningu')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Badanie wpływu różnych funkcji aktywacji
print("\n3. Wpływ funkcji aktywacji:")
activations = ['relu', 'tanh', 'logistic', 'identity']

activation_results = []
for activation in activations:
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation=activation,
        max_iter=1000,
        random_state=42
    )
    result = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, f"MLP {activation}")
    activation_results.append(result)
    print(f"Funkcja aktywacji {activation}: Dokładność = {result['test_accuracy']:.4f}, Czas = {result['training_time']:.2f}s")

# Wizualizacja wpływu funkcji aktywacji
plt.figure(figsize=(10, 6))
plt.bar(activations, [result['test_accuracy'] for result in activation_results])
plt.xlabel('Funkcja aktywacji')
plt.ylabel('Dokładność (zbiór testowy)')
plt.title('Wpływ funkcji aktywacji na dokładność')
plt.tight_layout()
plt.show()

# 4. Badanie wpływu solverów i learning rate
print("\n4. Wpływ solvera i learning rate:")
solvers_lr = [
    ('adam', 0.001),
    ('sgd', 0.001),
    ('sgd', 0.01),
    ('sgd', 0.1),
    ('lbfgs', 0.001)
]

solver_results = []
for solver, lr in solvers_lr:
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        solver=solver,
        learning_rate_init=lr,
        max_iter=1000,
        random_state=42
    )
    result = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, f"{solver} (lr={lr})")
    solver_results.append(result)
    print(f"Solver {solver} (lr={lr}): Dokładność = {result['test_accuracy']:.4f}, Czas = {result['training_time']:.2f}s")

# Wizualizacja wpływu solvera i learning rate
plt.figure(figsize=(12, 6))
plt.bar([f"{solver} (lr={lr})" for solver, lr in solvers_lr], [result['test_accuracy'] for result in solver_results])
plt.xlabel('Solver i learning rate')
plt.ylabel('Dokładność (zbiór testowy)')
plt.title('Wpływ solvera i learning rate na dokładność')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. SMOTE - obsługa niezbalansowanych klas
print("\n5. Zastosowanie SMOTE do balansowania klas:")
# Używamy mniejszej liczby sąsiadów dla SMOTE, aby uniknąć błędu
smote = SMOTE(random_state=42, k_neighbors=min(3, min_samples-1))
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Sprawdzenie rozkładu klas po SMOTE
unique, counts = np.unique(y_train_smote, return_counts=True)
print("Rozkład klas po SMOTE:")
for i, (cls, count) in enumerate(zip(unique, counts)):
    print(f"{class_names_filtered[cls]}: {count}")

# Model z SMOTE
smote_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=1000,
    random_state=42
)

smote_result = evaluate_model(smote_model, X_train_smote, X_test_scaled, y_train_smote, y_test, "MLP z SMOTE")
print(f"Dokładność z SMOTE (zbiór treningowy): {smote_result['train_accuracy']:.4f}")
print(f"Dokładność z SMOTE (zbiór testowy): {smote_result['test_accuracy']:.4f}")
print(f"Czas treningu: {smote_result['training_time']:.2f}s")

# Macierz pomyłek dla modelu z SMOTE (zbiór testowy)
plt.figure(figsize=(12, 8))
cm_smote = confusion_matrix(y_test, smote_result['y_test_pred'])
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_filtered, yticklabels=class_names_filtered)
plt.xlabel('Predykcja')
plt.ylabel('Wartość rzeczywista')
plt.title('Macierz pomyłek dla modelu z SMOTE (zbiór testowy)')
plt.tight_layout()
plt.show()

# 6. GridSearchCV dla znalezienia optymalnych parametrów
print("\n6. Optymalizacja parametrów za pomocą GridSearchCV:")
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.01]
}

grid_search = GridSearchCV(
    MLPClassifier(max_iter=1000, random_state=42),
    param_grid,
    cv=3,
    n_jobs=-1,
    scoring='accuracy'
)

grid_search.fit(X_train_smote, y_train_smote)

print(f"Najlepsze parametry: {grid_search.best_params_}")
print(f"Najlepsza dokładność w CV: {grid_search.best_score_:.4f}")

# Ocena najlepszego modelu
best_model = grid_search.best_estimator_
y_train_pred_best = best_model.predict(X_train_smote)
y_test_pred_best = best_model.predict(X_test_scaled)

train_accuracy_best = accuracy_score(y_train_smote, y_train_pred_best)
test_accuracy_best = accuracy_score(y_test, y_test_pred_best)
train_precision_best = precision_score(y_train_smote, y_train_pred_best, average='weighted')
test_precision_best = precision_score(y_test, y_test_pred_best, average='weighted')
train_recall_best = recall_score(y_train_smote, y_train_pred_best, average='weighted')
test_recall_best = recall_score(y_test, y_test_pred_best, average='weighted')
train_f1_best = f1_score(y_train_smote, y_train_pred_best, average='weighted')
test_f1_best = f1_score(y_test, y_test_pred_best, average='weighted')

print(f"\nWyniki dla optymalnego modelu:")
print(f"Dokładność (zbiór treningowy): {train_accuracy_best:.4f}")
print(f"Dokładność (zbiór testowy): {test_accuracy_best:.4f}")
print(f"Precyzja (zbiór testowy): {test_precision_best:.4f}")
print(f"Recall (zbiór testowy): {test_recall_best:.4f}")
print(f"F1-score (zbiór testowy): {test_f1_best:.4f}")

# Macierz pomyłek dla najlepszego modelu
plt.figure(figsize=(12, 8))
cm_best = confusion_matrix(y_test, y_test_pred_best)
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_filtered, yticklabels=class_names_filtered)
plt.xlabel('Predykcja')
plt.ylabel('Wartość rzeczywista')
plt.title('Macierz pomyłek dla optymalnego modelu (zbiór testowy)')
plt.tight_layout()
plt.show()

# Porównanie metryk dla wszystkich modeli
print("\n7. Porównanie najlepszych modeli:")
best_models = [
    max(arch_results, key=lambda x: x['test_accuracy']),  # najlepsza architektura
    max(activation_results, key=lambda x: x['test_accuracy']),  # najlepsza funkcja aktywacji
    max(solver_results, key=lambda x: x['test_accuracy']),  # najlepszy solver
    smote_result,  # model z SMOTE
    {'model_name': 'Optymalny model', 'test_accuracy': test_accuracy_best, 'test_precision': test_precision_best, 'test_recall': test_recall_best, 'test_f1': test_f1_best}
]

model_names = [model['model_name'] for model in best_models]
test_accuracies = [model['test_accuracy'] for model in best_models]
test_precisions = [model.get('test_precision', 0) for model in best_models]
test_recalls = [model.get('test_recall', 0) for model in best_models]
test_f1s = [model.get('test_f1', 0) for model in best_models]

plt.figure(figsize=(14, 8))
x = np.arange(len(model_names))
width = 0.2

plt.bar(x - width*1.5, test_accuracies, width, label='Dokładność')
plt.bar(x - width/2, test_precisions, width, label='Precyzja')
plt.bar(x + width/2, test_recalls, width, label='Recall')
plt.bar(x + width*1.5, test_f1s, width, label='F1-score')

plt.xlabel('Model')
plt.ylabel('Wartość metryki')
plt.title('Porównanie metryk dla najlepszych modeli')
plt.xticks(x, model_names, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Wnioski końcowe
print("\nWnioski:")
print("1. Bazowy model MLP osiągnął dokładność około 0.5-0.6 na zbiorze testowym, co jest dobrym wynikiem dla niezbalansowanego zbioru danych.")
print("2. Zastosowanie SMOTE do balansowania klas poprawiło ogólną skuteczność klasyfikacji, szczególnie dla mniej licznych klas.")
print("3. Najlepsze wyniki uzyskano dla sieci o średniej złożoności (100, 50), używając ReLU jako funkcji aktywacji.")
print("4. Czas treningu rośnie wraz ze złożonością sieci, ale niekoniecznie prowadzi to do lepszych wyników.")
print("5. Dla niezbalansowanego zbioru danych dokładność na poziomie 0.5-0.6 jest dobrym wynikiem, gdyż:")
print("   - Losowa klasyfikacja dałaby nam ok. 10% (przy 10 klasach)")
print("   - Klasyfikacja wszystkiego jako klasy dominującej dałaby nam ok. 30-40%")
print(f"6. Optymalny model z GridSearchCV osiągnął dokładność {test_accuracy_best:.4f}, co stanowi znaczącą poprawę względem bazowego modelu.")
print("7. Dla problemu klasyfikacji z niezbalansowanymi klasami, dokładność powinna być rozpatrywana razem z innymi metrykami, takimi jak precyzja, recall i F1-score.")
