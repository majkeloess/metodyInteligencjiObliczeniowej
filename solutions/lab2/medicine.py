import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = pd.read_csv('./medicine.txt')

print("Struktura danych:")
print(data.head())
print("\nPodsumowanie statystyczne:")
print(data.describe())
print("\nLiczba próbek w klasach:")
print(data['Was medicine effective?'].value_counts())

# Wykrywanie outlierów metodą IQR (Interquartile Range)
X = data.iloc[:, 0:2]
y = data.iloc[:, 2]

# Wizualizacja danych przed usunięciem outlierów
plt.figure(figsize=(10, 8))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
plt.colorbar()
plt.xlabel('Stężenie składnika 1')
plt.ylabel('Stężenie składnika 2')
plt.title('Dane przed usunięciem outlierów')
plt.show()

def detect_outliers_iqr(df, cols, factor=1.5):
    outlier_indices = []
    
    for col in cols:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_low = Q1 - factor * IQR
        outlier_high = Q3 + factor * IQR
        
        outliers = df[(df[col] < outlier_low) | (df[col] > outlier_high)].index
        outlier_indices.extend(outliers)
    
    return list(set(outlier_indices))

cols = X.columns.tolist()
outlier_indices = detect_outliers_iqr(X, cols)
print(f"\nLiczba wykrytych outlierów: {len(outlier_indices)}")

X_clean = X.drop(outlier_indices)
y_clean = y.drop(outlier_indices)

print(f"Liczba próbek po usunięciu outlierów: {X_clean.shape[0]}")
print(f"Liczba próbek w klasach po usunięciu outlierów: {y_clean.value_counts()}")

plt.figure(figsize=(10, 8))
plt.scatter(X_clean.iloc[:, 0], X_clean.iloc[:, 1], c=y_clean, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
plt.colorbar()
plt.xlabel('Stężenie składnika 1')
plt.ylabel('Stężenie składnika 2')
plt.title('Dane po usunięciu outlierów')
plt.show()

X = X_clean.values
y = y_clean.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nRozmiar zbioru treningowego: {X_train.shape[0]}")
print(f"Rozmiar zbioru testowego: {X_test.shape[0]}")
print(f"Proporcje klas w zbiorze treningowym: {np.bincount(y_train.astype(int))}")
print(f"Proporcje klas w zbiorze testowym: {np.bincount(y_test.astype(int))}")

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def visualize_results(clf, X, y, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
    
    xx, yy = make_meshgrid(X[:, 0], X[:, 1])
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.3)
    
    ax.set_xlabel('Stężenie składnika 1 (znormalizowane)')
    ax.set_ylabel('Stężenie składnika 2 (znormalizowane)')
    ax.set_title(title)
    plt.colorbar(scatter)
    
    plt.show()

network_structures = [
    {'hidden_layer_sizes': (5,), 'name': '1 warstwa, 5 neuronów'},
    {'hidden_layer_sizes': (10,), 'name': '1 warstwa, 10 neuronów'},
    {'hidden_layer_sizes': (5, 5), 'name': '2 warstwy, 5 neuronów każda'},
    {'hidden_layer_sizes': (10, 10), 'name': '2 warstwy, 10 neuronów każda'},
    {'hidden_layer_sizes': (20, 10, 5), 'name': '3 warstwy, 20-10-5 neuronów'},
    {'hidden_layer_sizes': (15, 15, 15), 'name': '3 warstwy, 15 neuronów każda'},
    {'hidden_layer_sizes': (50, 25, 10, 5), 'name': '4 warstwy, 50-25-10-5 neuronów'},
    {'hidden_layer_sizes': (30, 20, 10, 5), 'name': '4 warstwy, 30-20-10-5 neuronów'}
]

results = []

for struct in network_structures:
    print(f"\nTestowanie struktury: {struct['name']}")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=struct['hidden_layer_sizes'],
        max_iter=1000,
        random_state=42,
        solver='adam'
    )
    
    mlp.fit(X_train, y_train)
    
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Dokładność: {accuracy:.4f}")
    print("Raport klasyfikacji:")
    print(classification_report(y_test, y_pred))
    
    results.append({
        'structure': struct['name'],
        'accuracy': accuracy,
        'model': mlp
    })
    
    visualize_results(mlp, X_test, y_test, f"Granice decyzyjne: {struct['name']}")

results.sort(key=lambda x: x['accuracy'], reverse=True)

print("\n=== PODSUMOWANIE WYNIKÓW ===")
for i, result in enumerate(results):
    print(f"{i+1}. {result['structure']} - Dokładność: {result['accuracy']:.4f}")

best_model = results[0]['model']
worst_model = results[-1]['model']

print("\nWizualizacja dla najlepszej sieci:")
visualize_results(best_model, X_scaled, y, f"Najlepsza sieć: {results[0]['structure']}")

print("\nWizualizacja dla najgorszej sieci:")
visualize_results(worst_model, X_scaled, y, f"Najgorsza sieć: {results[-1]['structure']}")

print("\nPrawdziwy podział danych:")
plt.figure(figsize=(10, 8))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
plt.colorbar()
plt.xlabel('Stężenie składnika 1 (znormalizowane)')
plt.ylabel('Stężenie składnika 2 (znormalizowane)')
plt.title('Prawdziwy podział danych')
plt.show() 