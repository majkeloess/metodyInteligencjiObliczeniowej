from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, accuracy_score
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

# Analiza dla różnych inicjalizacji perceptronu
random_states = [42, 123, 256, 789, 101]
results = []

print("Analiza wielokrotnych uruchomień perceptronu:")
print("-" * 50)

for rs in random_states:
    model = Perceptron(max_iter=1000, random_state=rs)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    results.append({
        'random_state': rs,
        'accuracy': acc,
        'confusion_matrix': cm
    })
    
    print(f"\nRandom state: {rs}")
    print(f"Dokładność: {acc:.2%}")
    print("Macierz pomyłek:")
    print(cm)

# Średnia dokładność ze wszystkich uruchomień
avg_accuracy = np.mean([r['accuracy'] for r in results])
print("\nŚrednia dokładność ze wszystkich uruchomień: {:.2%}".format(avg_accuracy))

# Wizualizacja macierzy pomyłek dla najlepszego wyniku
best_result = max(results, key=lambda x: x['accuracy'])
print(f"\nNajlepszy wynik (random_state={best_result['random_state']}): {best_result['accuracy']:.2%}")

plt.figure(figsize=(8, 6))
sns.heatmap(best_result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title(f'Macierz pomyłek (dokładność: {best_result["accuracy"]:.2%})')
plt.xlabel('Predykcja')
plt.ylabel('Wartość rzeczywista')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

# Analiza liniowej separowalności danych Iris
print("\n\nAnaliza liniowej separowalności:")
print("-" * 50)
print("Dla zbioru Iris znany jest fakt, że klasa 'setosa' jest liniowo separowalna")
print("od pozostałych dwóch klas, ale klasy 'versicolor' i 'virginica' nie są względem siebie")
print("liniowo separowalne. Z tego powodu pojedyncza warstwa perceptronów ma problem z osiągnięciem")
print("wysokiej dokładności dla wszystkich trzech klas.\n")

# Sprawdźmy dokładność dla poszczególnych klas w najlepszym modelu
best_cm = best_result['confusion_matrix']
class_accuracies = [best_cm[i,i]/np.sum(best_cm[i,:]) for i in range(len(iris.target_names))]

for i, name in enumerate(iris.target_names):
    print(f"Dokładność dla klasy '{name}': {class_accuracies[i]:.2%}")

print("\nWnioski:")
print("Maksymalna dokładność, jaką może osiągnąć pojedyncza warstwa perceptronów dla zbioru Iris,")
print("jest ograniczona przez brak liniowej separowalności między klasami 'versicolor' i 'virginica'.")
print("Aby osiągnąć wyższą dokładność, potrzebne byłyby bardziej złożone modele, takie jak perceptron")
print("wielowarstwowy (MLP) lub inne modele nieliniowe, które potrafią lepiej radzić sobie")
print("z danymi, które nie są liniowo separowalne.")

# Wizualizacja problemu liniowej separowalności na dwóch wymiarach
plt.figure(figsize=(12, 5))

# Wybierzmy dwie cechy do wizualizacji
features = [(0, 1), (2, 3)]
titles = ['Długość i szerokość działki kielicha', 'Długość i szerokość płatka']

for i, (f1, f2) in enumerate(features):
    plt.subplot(1, 2, i+1)
    for target, color in zip(range(3), ['blue', 'red', 'green']):
        plt.scatter(X_test[y_test == target, f1], X_test[y_test == target, f2], c=color, 
                    label=iris.target_names[target], alpha=0.7)
    
    plt.xlabel(iris.feature_names[f1])
    plt.ylabel(iris.feature_names[f2])
    plt.legend()
    plt.title(titles[i])

plt.tight_layout()
plt.savefig('feature_visualization.png')