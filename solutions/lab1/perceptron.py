import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


K1 = np.random.normal(loc=[0,-1],scale=1.0, size=(100,2))
K2 = np.random.normal(loc=[1,1],scale=1.0, size=(100,2))

learning_size = [5,10,20, 100]

X = np.vstack((K1, K2))
y = np.hstack((np.zeros(100), np.ones(100)))


train_sizes = [5, 10, 20, 100]
datasets = []

for size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=size,
        stratify=y,
        random_state=42
    )
    datasets.append((X_train, X_test, y_train, y_test))

def find_decision_boundary(model):
    w = model.coef_[0]
    b = model.intercept_[0]
    # Równanie prostej: w0*x + w1*y + b = 0
    return lambda x: (-w[0]/w[1])*x - b/w[1]

for (X_train, X_test, y_train, y_test) in datasets:
    model = Perceptron(max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)
    boundary_line = find_decision_boundary(model)

def plot_results(X, y, boundary, title):
    plt.scatter(X[y==0][:,0], X[y==0][:,1], c='red', label='K1')
    plt.scatter(X[y==1][:,0], X[y==1][:,1], c='blue', label='K2')

    x_vals = np.array([X[:,0].min(), X[:,0].max()])
    y_vals = boundary(x_vals)
    plt.plot(x_vals, y_vals, '--k', label='Granica decyzyjna')

    plt.title(title)
    plt.legend()


for i, (X_train, X_test, y_train, y_test) in enumerate(datasets):
    plot_results(X_test, y_test, boundary_line,  f"{train_sizes[i]} próbek treningowych")
    plt.show()

for i, (X_train, X_test, y_train, y_test) in enumerate(datasets):
    model = Perceptron(max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    print(f"Rozmiar treningowy: {train_sizes[i]}")
    print(f"Dokładność: {acc:.2f}")
    print(f"Równanie prostej: y = {model.coef_[0][0]/model.coef_[0][1]:.2f}x + {model.intercept_[0]/model.coef_[0][1]:.2f}")
    print("-------------------")