import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Definicja funkcji celu
def objective_function(x, y):
    term1 = 2 * np.log(np.abs(x + 0.2) + 0.002)
    term2 = np.log(np.abs(y + 0.1) + 0.001)
    term3 = np.cos(3 * x)
    term4 = 2 * (np.sin(3 * x * y)**2)
    term5 = np.sin(y)**2
    term6 = -x**2
    term7 = -0.5 * y**2
    return term1 + term2 + term3 + term4 + term5 + term6 + term7

# Implementacja algorytmu PSO
def pso(n_particles, n_dimensions, bounds, n_iterations, c1, c2, w, w_strategy='constant', v_max_ratio=0.2, random_seed=None):
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
    
    convergence_history = [gbest_fitness]

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
        particles_vel = np.clip(particles_vel, -v_max, v_max) # Ograniczenie prędkości
        
        particles_pos = particles_pos + particles_vel
        particles_pos = np.clip(particles_pos, min_bound, max_bound) # Ograniczenie pozycji
        
        current_fitness = np.array([objective_function(p[0], p[1]) for p in particles_pos])
        
        improved_indices = current_fitness < pbest_fitness
        pbest_pos[improved_indices] = particles_pos[improved_indices]
        pbest_fitness[improved_indices] = current_fitness[improved_indices]
        
        current_best_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_best_idx] < gbest_fitness:
            gbest_pos = np.copy(pbest_pos[current_best_idx])
            gbest_fitness = pbest_fitness[current_best_idx]
            
        convergence_history.append(gbest_fitness)
        
    return gbest_pos, gbest_fitness, convergence_history

def run_experiment(n_runs, config_name, **pso_params):
    print(f"--- Rozpoczęcie eksperymentu: {config_name} ---")
    all_fitness_values = []
    
    # Pojedynczy przebieg dla wykresu
    plot_seed = 42 
    _, _, single_run_history = pso(**pso_params, random_seed=plot_seed)
    
    plt.figure(figsize=(10, 6))
    plt.plot(single_run_history)
    plt.title(f'Zbieżność PSO dla {config_name} (jeden przebieg)')
    plt.xlabel('Iteracja')
    plt.ylabel('Najlepsza wartość funkcji celu')
    plt.grid(True)
    plt.show()

    # 10 przebiegów dla statystyk
    for i in range(n_runs):
        _, best_fitness, _ = pso(**pso_params, random_seed=i) # Różne ziarna dla każdego przebiegu
        all_fitness_values.append(best_fitness)
    
    mean_fitness = np.mean(all_fitness_values)
    std_fitness = np.std(all_fitness_values)
    
    print(f"Wyniki dla {config_name}:")
    print(f"  Średnia wartość funkcji przystosowania: {mean_fitness:.6f}")
    print(f"  Odchylenie standardowe: {std_fitness:.6f}")
    print(f"--- Koniec eksperymentu: {config_name} ---\\n")
    return mean_fitness, std_fitness, config_name


if __name__ == "__main__":
    N_PARTICLES = 30
    N_DIMENSIONS = 2
    BOUNDS = ([-1.0, -1.0], [1.0, 1.0])
    N_ITERATIONS = 100
    N_RUNS = 10

    # Tworzenie katalogu na wykresy, jeśli nie istnieje
    # plot_directory = "pso_plots_zad1"
    # if not os.path.exists(plot_directory):
    #     os.makedirs(plot_directory)

    all_results = []

    # Definicja konfiguracji eksperymentów
    # Ustalona wartość w dla testów c1, c2
    fixed_w_for_c_tests = 0.7

    experiments_c1_c2 = [
        {"c1": 0, "c2": 2, "w": fixed_w_for_c_tests, "w_strategy": "constant", "name": "c1=0_c2=2_w=0.7"},
        {"c1": 2, "c2": 0, "w": fixed_w_for_c_tests, "w_strategy": "constant", "name": "c1=2_c2=0_w=0.7"},
        {"c1": 1, "c2": 1, "w": fixed_w_for_c_tests, "w_strategy": "constant", "name": "c1=1_c2=1_w=0.7"},
        {"c1": 0.5, "c2": 1.5, "w": fixed_w_for_c_tests, "w_strategy": "constant", "name": "c1=0.5_c2=1.5_w=0.7"},
        {"c1": 1.5, "c2": 0.5, "w": fixed_w_for_c_tests, "w_strategy": "constant", "name": "c1=1.5_c2=0.5_w=0.7"},
        {"c1": 2.2, "c2": 2.2, "w": fixed_w_for_c_tests, "w_strategy": "constant", "name": "c1=2.2_c2=2.2_w=0.7"},
    ]

    print("Rozpoczęcie badań wpływu parametrów c1 i c2 (w=0.7):")
    for exp_config in experiments_c1_c2:
        params = {
            "n_particles": N_PARTICLES, "n_dimensions": N_DIMENSIONS,
            "bounds": BOUNDS, "n_iterations": N_ITERATIONS,
            "c1": exp_config["c1"], "c2": exp_config["c2"],
            "w": exp_config["w"], "w_strategy": exp_config["w_strategy"]
        }
        mean_f, std_f, name = run_experiment(N_RUNS, exp_config["name"], **params)
        all_results.append({"name": name, "mean_fitness": mean_f, "std_fitness": std_f, 
                            "c1": exp_config["c1"], "c2": exp_config["c2"], "w_val": exp_config["w"], "w_strat": exp_config["w_strategy"]})

    # Ustalona para c1, c2 dla testów w
    # Wybierzmy c1=1.5, c2=1.5 jako bazowe, lub c1=c2=2.2, które było też testowane
    # Dla spójności z poleceniem, można użyć c1=c2=2.2 jeśli wypadło dobrze, 
    # lub standardowe c1=1.5, c2=1.5
    # Użyjmy c1=1.5, c2=1.5 dla testów 'w'
    fixed_c1_for_w_tests = 1.5
    fixed_c2_for_w_tests = 1.5
    
    print("\\nRozpoczęcie badań wpływu parametru w (c1=1.5, c2=1.5):")
    experiments_w = [
        {"w": 0.4, "w_strategy": "constant", "name": "w=0.4_c1=1.5_c2=1.5"},
        {"w": 0.7, "w_strategy": "constant", "name": "w=0.7_c1=1.5_c2=1.5"},
        {"w": 0.9, "w_strategy": "constant", "name": "w=0.9_c1=1.5_c2=1.5"},
        {"w": (0.9, 0.4), "w_strategy": "linear_decrease", "name": "w=0.9-0.4_c1=1.5_c2=1.5"},
    ]
    
    for exp_config in experiments_w:
        params = {
            "n_particles": N_PARTICLES, "n_dimensions": N_DIMENSIONS,
            "bounds": BOUNDS, "n_iterations": N_ITERATIONS,
            "c1": fixed_c1_for_w_tests, "c2": fixed_c2_for_w_tests,
            "w": exp_config["w"], "w_strategy": exp_config["w_strategy"]
        }
        mean_f, std_f, name = run_experiment(N_RUNS, exp_config["name"], **params)
        all_results.append({"name": name, "mean_fitness": mean_f, "std_fitness": std_f,
                            "c1": fixed_c1_for_w_tests, "c2": fixed_c2_for_w_tests, "w_val": exp_config["w"], "w_strat": exp_config["w_strategy"]})

    print("\\n--- Podsumowanie wszystkich eksperymentów ---")
    # Sortowanie wyników: najpierw według średniej wartości funkcji (im niższa, tym lepiej),
    # a potem według odchylenia standardowego (im niższe, tym lepiej)
    all_results.sort(key=lambda r: (r["mean_fitness"], r["std_fitness"]))

    for res in all_results:
        w_info = f"w={res['w_val']}" if res['w_strat'] == 'constant' else f"w={res['w_val'][0]}-{res['w_val'][1]} ({res['w_strat']})"
        print(f"Konfiguracja: c1={res['c1']}, c2={res['c2']}, {w_info} -> Średnia: {res['mean_fitness']:.6f}, StdDev: {res['std_fitness']:.6f}")

    if all_results:
        best_config = all_results[0]
        w_info_best = f"w={best_config['w_val']}" if best_config['w_strat'] == 'constant' else f"w={best_config['w_val'][0]}-{best_config['w_val'][1]} ({best_config['w_strat']})"
        print(f"\\nNajlepsza konfiguracja (najniższa średnia wartość funkcji celu, następnie najniższe odchylenie standardowe):")
        print(f"  Nazwa: {best_config['name']}")
        print(f"  Parametry: c1={best_config['c1']}, c2={best_config['c2']}, {w_info_best}")
        print(f"  Średnia wartość funkcji celu: {best_config['mean_fitness']:.6f}")
        print(f"  Odchylenie standardowe: {best_config['std_fitness']:.6f}")
    else:
        print("Nie przeprowadzono żadnych eksperymentów.")

