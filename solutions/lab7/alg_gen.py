import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Callable, List, Tuple, Dict, Any
from enum import Enum

class Encoding(Enum):
    BINARY = "binary"
    GRAY = "gray"

class Selection(Enum):
    ROULETTE = "roulette"
    THRESHOLD = "threshold"

# Funkcja celu
def objective_function(x: float) -> float:
    """Funkcja celu do maksymalizacji."""
    if x == 0:
        return 0
    return np.cos(80 * x + 0.3) + 3 * x - 0.9 - 2

# Konwersja liczby dziesiętnej na binarną o określonej długości
def decimal_to_binary(n: int, bits: int) -> str:
    """Konwersja liczby dziesiętnej na binarną o określonej długości."""
    return format(n, f'0{bits}b')

# Konwersja binarnej na kod Graya
def binary_to_gray(binary: str) -> str:
    gray = binary[0]
    for i in range(1, len(binary)):
        gray += str(int(binary[i-1]) ^ int(binary[i]))
    return gray

# Konwersja kodu Graya na binarne
def gray_to_binary(gray: str) -> str:
    binary = gray[0]
    for i in range(1, len(gray)):
        binary += str(int(binary[i-1]) ^ int(gray[i]))
    return binary

# Kodowanie wartości rzeczywistej na łańcuch bitów
def encode(x: float, min_val: float, max_val: float, bits: int, encoding: Encoding) -> str:
    """Kodowanie wartości rzeczywistej na łańcuch bitów."""
    value_range = max_val - min_val
    normalized = (x - min_val) / value_range
    decimal = int(normalized * (2**bits - 1))
    binary = decimal_to_binary(decimal, bits)
    
    if encoding == Encoding.GRAY:
        return binary_to_gray(binary)
    return binary

# Dekodowanie łańcucha bitów na wartość rzeczywistą
def decode(chromosome: str, min_val: float, max_val: float, encoding: Encoding) -> float:
    """Dekodowanie łańcucha bitów na wartość rzeczywistą."""
    binary = chromosome
    if encoding == Encoding.GRAY:
        binary = gray_to_binary(chromosome)
    
    decimal = int(binary, 2)
    value_range = max_val - min_val
    normalized = decimal / (2**len(chromosome) - 1)
    return min_val + normalized * value_range

# Krzyżowanie jednopunktowe
def crossover(parent1: str, parent2: str) -> Tuple[str, str]:
    """Krzyżowanie jednopunktowe."""
    if len(parent1) != len(parent2):
        raise ValueError("Chromosomy rodziców muszą być tej samej długości")
    
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    return child1, child2

# Mutacja bitowa
def mutate(chromosome: str, mutation_rate: float) -> str:
    """Mutacja bitowa."""
    mutated = ""
    for bit in chromosome:
        if random.random() < mutation_rate:
            mutated += '1' if bit == '0' else '0'
        else:
            mutated += bit
    return mutated

# Selekcja ruletkowa
def roulette_selection(population: List[str], fitness_values: List[float], 
                       num_selected: int) -> List[str]:
    """Selekcja ruletkowa."""
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        # Jeśli wszystkie wartości fitness są zerowe, wybierz losowo
        return random.choices(population, k=num_selected)
    
    relative_fitness = [f / total_fitness for f in fitness_values]
    cumulative_probs = np.cumsum(relative_fitness)
    
    selected = []
    for _ in range(num_selected):
        r = random.random()
        for i, cp in enumerate(cumulative_probs):
            if r <= cp:
                selected.append(population[i])
                break
    
    return selected

# Selekcja progowa
def threshold_selection(population: List[str], fitness_values: List[float], gamma: float) -> List[str]:
    # Sortuj populację według wartości fitness (malejąco)
    sorted_indices = np.argsort(fitness_values)[::-1]
    sorted_population = [population[i] for i in sorted_indices]
    
    # Wybierz gamma% najlepszych osobników
    threshold_idx = max(1, int(len(population) * gamma / 100))
    top_individuals = sorted_population[:threshold_idx]
    
    # Wypełnij nową populację losowymi osobnikami z najlepszych
    return random.choices(top_individuals, k=len(population))

# Algorytm genetyczny
def genetic_algorithm(
    objective_func: Callable[[float], float],
    min_val: float,
    max_val: float,
    encoding_type: Encoding,
    selection_type: Selection,
    mutation_rate: float,
    population_size: int,
    num_generations: int,
    chromosome_length: int,
    gamma: float = 30
) -> Dict[str, Any]:
    
    # Inicjalizacja populacji
    population = [''.join(random.choice(['0', '1']) for _ in range(chromosome_length)) 
                 for _ in range(population_size)]
    
    best_individual = None
    best_fitness = float('-inf')
    fitness_history = []
    best_individual_history = []
    
    for generation in range(num_generations):
        # Dekodowanie i obliczanie fitness
        decoded_values = [decode(chrom, min_val, max_val, encoding_type) for chrom in population]
        fitness_values = [objective_func(val) for val in decoded_values]
        
        # Znajdź najlepszego osobnika
        gen_best_idx = np.argmax(fitness_values)
        gen_best_fitness = fitness_values[gen_best_idx]
        gen_best_individual = decoded_values[gen_best_idx]
        
        # Aktualizuj najlepszego osobnika ogólnie
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = gen_best_individual
            
        # Zapisz historię
        fitness_history.append(np.mean(fitness_values))
        best_individual_history.append(gen_best_fitness)
        
        # Selekcja
        if selection_type == Selection.ROULETTE:
            selected = roulette_selection(population, fitness_values, population_size)
        else:  # Threshold selection
            selected = threshold_selection(population, fitness_values, gamma)
        
        # Przygotuj nową populację
        new_population = []
        
        # Elityzm - zachowaj najlepszego osobnika
        new_population.append(population[gen_best_idx])
        
        # Krzyżowanie i mutacja
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = crossover(parent1, parent2)
            
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
        
        # Aktualizuj populację
        population = new_population
    
    return {
        'best_individual': best_individual,
        'best_fitness': best_fitness,
        'fitness_history': fitness_history,
        'best_individual_history': best_individual_history
    }

# Funkcja do przeprowadzenia eksperymentów
def run_experiments():
    # Parametry
    min_val = 0.01
    max_val = 1.0
    population_size = 50
    num_generations = 100
    chromosome_length = 16
    num_runs = 10
    
    # Konfiguracje
    encodings = [Encoding.BINARY, Encoding.GRAY]
    mutation_rates = [0, 0.1, 0.5, 1.0]
    selections = [
        (Selection.ROULETTE, 0),
        (Selection.THRESHOLD, 30),
        (Selection.THRESHOLD, 60)
    ]
    
    results = {}
    
    # Przeprowadź eksperymenty
    for encoding in encodings:
        for mutation_rate in mutation_rates:
            for selection, gamma in selections:
                config_name = f"{encoding.value}_{mutation_rate}_{selection.value}"
                if selection == Selection.THRESHOLD:
                    config_name += f"_{gamma}"
                
                print(f"Uruchamiam konfigurację: {config_name}")
                
                config_results = []
                example_run = None
                
                for run in range(num_runs):
                    result = genetic_algorithm(
                        objective_func=objective_function,
                        min_val=min_val,
                        max_val=max_val,
                        encoding_type=encoding,
                        selection_type=selection,
                        mutation_rate=mutation_rate,
                        population_size=population_size,
                        num_generations=num_generations,
                        chromosome_length=chromosome_length,
                        gamma=gamma
                    )
                    
                    config_results.append(result['best_fitness'])
                    
                    # Zapisz pierwszy przebieg jako przykładowy
                    if run == 0:
                        example_run = result
                
                avg_fitness = sum(config_results) / len(config_results)
                results[config_name] = {
                    'avg_fitness': avg_fitness,
                    'example_run': example_run
                }
                
                print(f"  Średni wynik: {avg_fitness}")
    
    return results

# Funkcja do wyświetlania wyników
def plot_results(results):
    # Tabela wyników
    print("\nŚrednie wyniki dla 10 uruchomień:")
    print("=" * 70)
    print(f"{'Konfiguracja':<40} | {'Średni najlepszy wynik':<20}")
    print("=" * 70)
    
    # Sortowanie wyników według średniego fitness (malejąco)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_fitness'], reverse=True)
    
    # Wyświetl wszystkie wyniki w tabeli, ale posortowane
    for config, data in sorted_results:
        print(f"{config:<40} | {data['avg_fitness']:<20.6f}")
    
    # Wybierz tylko najlepszą konfigurację do wykresu
    best_config, best_data = sorted_results[0]
    print(f"\nNajlepsza konfiguracja: {best_config} (średni fitness: {best_data['avg_fitness']:.6f})")
    
    # Wykres historii najlepszej konfiguracji
    plt.figure(figsize=(12, 6))
    example_run = best_data['example_run']
    
    plt.plot(example_run['fitness_history'], label='Średni fitness')
    plt.plot(example_run['best_individual_history'], label='Najlepszy fitness')
    plt.title(f"Najlepsza konfiguracja: {best_config}")
    plt.xlabel('Generacja')
    plt.ylabel('Wartość funkcji celu')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Wykres funkcji celu z zaznaczonym najlepszym znalezionym maksimum
    x = np.linspace(0.01, 1, 1000)
    y = [objective_function(xi) for xi in x]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    
    # Zaznacz tylko najlepsze znalezione maksimum
    best_x = best_data['example_run']['best_individual']
    best_y = best_data['example_run']['best_fitness']
    plt.scatter(best_x, best_y, color='red', s=100, label=f"Najlepsze: x={best_x:.4f}, f(x)={best_y:.4f}")
    
    plt.title('Funkcja celu i znalezione maksimum')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Uruchomienie eksperymentów
if __name__ == "__main__":
    results = run_experiments()
    plot_results(results)
