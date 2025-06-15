import numpy as np
import time
import os
import matplotlib.pyplot as plt # Dodane do rysowania, jeśli potrzebne

# --- Skopiowane z zad1.py ---
def objective_function(x, y):
    term1 = 2 * np.log(np.abs(x + 0.2) + 0.002)
    term2 = np.log(np.abs(y + 0.1) + 0.001)
    term3 = np.cos(3 * x)
    term4 = 2 * (np.sin(3 * x * y)**2)
    term5 = np.sin(y)**2
    term6 = -x**2
    term7 = -0.5 * y**2
    return term1 + term2 + term3 + term4 + term5 + term6 + term7

def pso_best_config(n_particles, n_dimensions, bounds, n_iterations, c1, c2, w, w_strategy='constant', v_max_ratio=0.2, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    min_bound, max_bound = np.array(bounds[0]), np.array(bounds[1])
    v_max = (max_bound - min_bound) * v_max_ratio

    particles_pos = np.random.uniform(min_bound, max_bound, (n_particles, n_dimensions))
    particles_vel = np.random.uniform(-v_max, v_max, (n_particles, n_dimensions))
    
    pbest_pos = np.copy(particles_pos)
    pbest_fitness = np.array([objective_function(p[0], p[1]) for p in pbest_pos])
    
    gbest_idx = np.argmin(pbest_fitness)
    gbest_pos = np.copy(pbest_pos[gbest_idx])
    gbest_fitness = pbest_fitness[gbest_idx]
    
    w_current = w
    if w_strategy == 'linear_decrease' and isinstance(w, tuple):
        w_start, w_end = w

    for iteration in range(n_iterations):
        if w_strategy == 'linear_decrease' and isinstance(w, tuple):
            w_current = w_start - (w_start - w_end) * (iteration / n_iterations)
        elif w_strategy == 'constant':
            w_current = w

        r1 = np.random.rand(n_particles, n_dimensions)
        r2 = np.random.rand(n_particles, n_dimensions)
        
        cognitive_component = c1 * r1 * (pbest_pos - particles_pos)
        social_component = c2 * r2 * (gbest_pos - particles_pos)
        
        particles_vel = w_current * particles_vel + cognitive_component + social_component
        particles_vel = np.clip(particles_vel, -v_max, v_max)
        particles_pos = particles_pos + particles_vel
        particles_pos = np.clip(particles_pos, min_bound, max_bound)
        
        current_fitness = np.array([objective_function(p[0], p[1]) for p in particles_pos])
        
        improved_indices = current_fitness < pbest_fitness
        pbest_pos[improved_indices] = particles_pos[improved_indices]
        pbest_fitness[improved_indices] = current_fitness[improved_indices]
        
        current_best_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_best_idx] < gbest_fitness:
            gbest_pos = np.copy(pbest_pos[current_best_idx])
            gbest_fitness = pbest_fitness[current_best_idx]
            
    return gbest_pos, gbest_fitness

def genetic_algorithm(population_size, n_dimensions, bounds, n_generations, mutation_rate, crossover_rate, tournament_size=3, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    min_bound, max_bound = np.array(bounds[0]), np.array(bounds[1])
    population = np.random.uniform(min_bound, max_bound, (population_size, n_dimensions))
    best_solution_overall = None
    best_fitness_overall = float('inf')

    for generation in range(n_generations):
        fitness_values = np.array([objective_function(ind[0], ind[1]) for ind in population])
        
        current_best_idx = np.argmin(fitness_values)
        if fitness_values[current_best_idx] < best_fitness_overall:
            best_fitness_overall = fitness_values[current_best_idx]
            best_solution_overall = np.copy(population[current_best_idx])

        selected_parents = []
        for _ in range(population_size):
            tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
            tournament_fitness = fitness_values[tournament_indices]
            winner_idx_in_tournament = np.argmin(tournament_fitness)
            selected_parents.append(population[tournament_indices[winner_idx_in_tournament]])
        selected_parents = np.array(selected_parents)

        offspring = []
        for i in range(0, population_size, 2):
            parent1 = selected_parents[i]
            parent2_idx = (i + 1) if (i+1) < population_size else i 
            parent2 = selected_parents[parent2_idx]
            
            child1, child2 = np.copy(parent1), np.copy(parent2)
            if np.random.rand() < crossover_rate:
                alpha = 0.5
                for j in range(n_dimensions):
                    d = np.abs(parent1[j] - parent2[j])
                    u = np.random.uniform(-alpha * d, (1 + alpha) * d)
                    child1[j] = parent1[j] + u
                    
                    u = np.random.uniform(-alpha * d, (1 + alpha) * d)
                    child2[j] = parent2[j] + u # Dla drugiego potomka, bazując na parent2 + u
                    # Lub można stworzyć drugiego potomka inaczej np. child2[j] = parent1[j] - u lub parent2[j] - u
                    # Dla uproszczenia, drugi potomek też może być podobnie generowany lub użyć innej strategii
                    # Tutaj dla prostoty, drugi potomek jest tworzony w ten sam sposób co pierwszy, ale z nowym 'u'

            offspring.extend([child1, child2])
        offspring = np.array(offspring[:population_size]) # Dopasuj rozmiar potomstwa

        # Mutacja (uniform mutation)
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(n_dimensions)
                offspring[i, mutation_point] = np.random.uniform(min_bound[mutation_point], max_bound[mutation_point])
        
        population = np.clip(offspring, min_bound, max_bound) # Ograniczenie do granic

    # Ostateczne obliczenie najlepszego rozwiązania po wszystkich generacjach
    final_fitness_values = np.array([objective_function(ind[0], ind[1]) for ind in population])
    final_best_idx = np.argmin(final_fitness_values)
    if final_fitness_values[final_best_idx] < best_fitness_overall:
        best_fitness_overall = final_fitness_values[final_best_idx]
        best_solution_overall = population[final_best_idx]
        
    return best_solution_overall, best_fitness_overall

if __name__ == "__main__":
    N_DIMENSIONS = 2
    BOUNDS = ([-1.0, -1.0], [1.0, 1.0])
    N_ITERATIONS_OR_GENERATIONS = 100 # Ta sama liczba dla obu algorytmów
    N_RUNS_COMPARISON = 10

    # --- Parametry dla najlepszej konfiguracji PSO (załóżmy, że to są te wartości) ---
    # Te wartości powinny być ustalone na podstawie wyników zad1.py
    # Przykład (należy zaktualizować po uruchomieniu zad1.py):
    BEST_PSO_PARAMS = {
        "n_particles": 30,
        "c1": 1.5, # Przykładowa wartość
        "c2": 1.5, # Przykładowa wartość
        "w": 0.7,  # Przykładowa wartość
        "w_strategy": 'constant', # lub 'linear_decrease' z krotką w
        "v_max_ratio": 0.2
    }
    print(f"UWAGA: Parametry najlepszej konfiguracji PSO (BEST_PSO_PARAMS) są przykładowe. ")
    print(f"Należy je zaktualizować na podstawie wyników z zad1.py! Aktualne: {BEST_PSO_PARAMS}\n")

    # --- Parametry dla GA ---
    GA_PARAMS = {
        "population_size": BEST_PSO_PARAMS.get("n_particles", 30), # Porównywalna wielkość populacji
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "tournament_size": 3
    }

    pso_fitness_results = []
    pso_time_results = []
    ga_fitness_results = []
    ga_time_results = []

    print(f"Rozpoczynanie porównania PSO vs GA ({N_RUNS_COMPARISON} przebiegów każdy, {N_ITERATIONS_OR_GENERATIONS} iteracji/generacji)")

    for i in range(N_RUNS_COMPARISON):
        print(f"  Przebieg {i+1}/{N_RUNS_COMPARISON}")
        current_seed = i # Użyjemy tego samego ziarna dla PSO i GA w danym przebiegu dla sprawiedliwości

        # Uruchomienie PSO
        start_time_pso = time.time()
        _, pso_fitness = pso_best_config(
            n_particles=BEST_PSO_PARAMS["n_particles"],
            n_dimensions=N_DIMENSIONS, 
            bounds=BOUNDS, 
            n_iterations=N_ITERATIONS_OR_GENERATIONS,
            c1=BEST_PSO_PARAMS["c1"], 
            c2=BEST_PSO_PARAMS["c2"], 
            w=BEST_PSO_PARAMS["w"], 
            w_strategy=BEST_PSO_PARAMS["w_strategy"],
            v_max_ratio=BEST_PSO_PARAMS["v_max_ratio"],
            random_seed=current_seed
        )
        end_time_pso = time.time()
        pso_fitness_results.append(pso_fitness)
        pso_time_results.append(end_time_pso - start_time_pso)

        # Uruchomienie GA
        start_time_ga = time.time()
        _, ga_fitness = genetic_algorithm(
            population_size=GA_PARAMS["population_size"],
            n_dimensions=N_DIMENSIONS,
            bounds=BOUNDS,
            n_generations=N_ITERATIONS_OR_GENERATIONS,
            mutation_rate=GA_PARAMS["mutation_rate"],
            crossover_rate=GA_PARAMS["crossover_rate"],
            tournament_size=GA_PARAMS["tournament_size"],
            random_seed=current_seed
        )
        end_time_ga = time.time()
        ga_fitness_results.append(ga_fitness)
        ga_time_results.append(end_time_ga - start_time_ga)

    # Obliczenie średnich wyników
    mean_pso_fitness = np.mean(pso_fitness_results)
    std_pso_fitness = np.std(pso_fitness_results)
    mean_pso_time = np.mean(pso_time_results)

    mean_ga_fitness = np.mean(ga_fitness_results)
    std_ga_fitness = np.std(ga_fitness_results)
    mean_ga_time = np.mean(ga_time_results)

    print("\n--- Wyniki Porównania ---")
    print("Algorytm Roju Cząstek (PSO) - Najlepsza Konfiguracja:")
    print(f"  Średnia wartość funkcji celu: {mean_pso_fitness:.6f} (StdDev: {std_pso_fitness:.6f})")
    print(f"  Średni czas wykonania: {mean_pso_time:.4f} s")

    print("\nAlgorytm Genetyczny (GA):")
    print(f"  Średnia wartość funkcji celu: {mean_ga_fitness:.6f} (StdDev: {std_ga_fitness:.6f})")
    print(f"  Średni czas wykonania: {mean_ga_time:.4f} s")

    print("\n--- Podsumowanie ---")
    if mean_pso_fitness < mean_ga_fitness:
        print("PSO uzyskało lepszą (niższą) średnią wartość funkcji celu.")
    elif mean_ga_fitness < mean_pso_fitness:
        print("Algorytm Genetyczny uzyskał lepszą (niższą) średnią wartość funkcji celu.")
    else:
        print("Oba algorytmy uzyskały podobną średnią wartość funkcji celu.")

    if mean_pso_time < mean_ga_time:
        print("PSO był szybszy.")
    elif mean_ga_time < mean_pso_time:
        print("Algorytm Genetyczny był szybszy.")
    else:
        print("Oba algorytmy miały podobny czas wykonania.")

    # Dodatkowe informacje o parametrach dla przejrzystości
    print("\nUżyte parametry PSO (najlepsza konfiguracja z zad1 - do weryfikacji!):")
    for key, value in BEST_PSO_PARAMS.items():
        print(f"  {key}: {value}")
    print("\nUżyte parametry GA:")
    for key, value in GA_PARAMS.items():
        print(f"  {key}: {value}")
    print(f"Liczba iteracji/generacji dla obu: {N_ITERATIONS_OR_GENERATIONS}")
    print(f"Liczba przebiegów do uśrednienia: {N_RUNS_COMPARISON}")
