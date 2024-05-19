import random
import numpy as np
import math
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import cv2
import os

# Función objetivo
def fitness(x):
    return x * np.cos(x)

# Inicialización de la población
def initialize_population(pop_size, num_bits):
    return [''.join(random.choice('01') for _ in range(num_bits)) for _ in range(pop_size)]

# Decodificación de un individuo
def decode(individual, num_bits, A, B):
    max_val = 2**num_bits - 1
    integer = int(individual, 2)
    return A + (integer / max_val) * (B - A)

# Evaluación de la población
def evaluate_population(population, num_bits, A, B):
    return [fitness(decode(ind, num_bits, A, B)) for ind in population]

# Selección de parejas
def select_pairs(population, n):
    pairs = []
    for i in range(len(population)):
        m = random.randint(0, n)
        mates = random.sample(range(len(population)), m)
        pairs.extend([(i, mate) for mate in mates if mate != i])
    return pairs

# Cruza
def crossover(pair, population, num_bits):
    p1, p2 = pair
    point = random.randint(1, num_bits - 1)
    offspring1 = population[p1][:point] + population[p2][point:]
    offspring2 = population[p2][:point] + population[p1][point:]
    return offspring1, offspring2

# Mutación
def mutate(individual, num_bits, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(num_bits), 2)
        individual = list(individual)
        individual[i], individual[j] = individual[j], individual[i]
        individual = ''.join(individual)
    return individual

# Poda
def prune_population(population, fitnesses, pop_size):
    best_individual = population[np.argmax(fitnesses)]
    indices = list(range(len(population)))
    indices.remove(np.argmax(fitnesses))
    to_keep = random.sample(indices, pop_size - 1)
    new_population = [population[i] for i in to_keep]
    new_population.append(best_individual)
    return new_population

# Función para maximizar la aptitud
def genetic_algorithm_maximize(A, B, delta_x, generations, mutation_rate_gene, pop_size, max_population, crossover_rate):
    num_bits = math.ceil(math.log2((B - A) / delta_x + 1))
    mutation_rate_initial = random.uniform(0, 1)  # Probabilidad de mutación inicial aleatoria
    
    population = initialize_population(pop_size, num_bits)
    
    best_fitnesses = []
    worst_fitnesses = []
    average_fitnesses = []
    
    if not os.path.exists('gen_images'):
        os.makedirs('gen_images')
    
    for generation in range(generations):
        fitnesses = evaluate_population(population, num_bits, A, B)
        
        best_fitness = max(fitnesses)
        worst_fitness = min(fitnesses)
        average_fitness = sum(fitnesses) / len(fitnesses)
        
        best_fitnesses.append(best_fitness)
        worst_fitnesses.append(worst_fitness)
        average_fitnesses.append(average_fitness)
        
        best_index = np.argmax(fitnesses)
        best_individual = population[best_index]
        best_value = decode(best_individual, num_bits, A, B)
        
        update_table(generation, best_individual, best_index, best_value, best_fitness)
        
        pairs = select_pairs(population, pop_size)
        offspring = []
        for pair in pairs:
            if random.random() < crossover_rate:
                off1, off2 = crossover(pair, population, num_bits)
                offspring.append(mutate(off1, num_bits, mutation_rate_gene))
                offspring.append(mutate(off2, num_bits, mutation_rate_gene))
        
        population.extend(offspring)
        
        fitnesses = evaluate_population(population, num_bits, A, B)
        population = prune_population(population, fitnesses, min(pop_size, max_population))

        plt.figure(figsize=(10, 6))
        plt.plot(best_fitnesses, label='Mejor aptitud')
        plt.plot(worst_fitnesses, label='Peor aptitud')
        plt.plot(average_fitnesses, label='Aptitud promedio')
        plt.xlabel('Generación')
        plt.ylabel('Aptitud')
        plt.title(f'Evolución de la Aptitud de la Población (Maximización) - Generación {generation}')
        plt.legend()
        plt.savefig(f'gen_images/gen_{generation}.png')
        plt.close()
    
    create_video('gen_images', 'evolution_maximization.mp4')

# Función para minimizar la aptitud
def genetic_algorithm_minimize(A, B, delta_x, generations, mutation_rate_gene, pop_size, max_population, crossover_rate):
    num_bits = math.ceil(math.log2((B - A) / delta_x + 1))
    mutation_rate_initial = random.uniform(0, 1)  # Probabilidad de mutación inicial aleatoria
    
    population = initialize_population(pop_size, num_bits)
    
    best_fitnesses = []
    worst_fitnesses = []
    average_fitnesses = []
    
    if not os.path.exists('gen_images'):
        os.makedirs('gen_images')
    
    for generation in range(generations):
        fitnesses = evaluate_population(population, num_bits, A, B)
        
        best_fitness = min(fitnesses)
        worst_fitness = max(fitnesses)
        average_fitness = sum(fitnesses) / len(fitnesses)
        
        best_fitnesses.append(best_fitness)
        worst_fitnesses.append(worst_fitness)
        average_fitnesses.append(average_fitness)
        
        best_index = np.argmin(fitnesses)
        best_individual = population[best_index]
        best_value = decode(best_individual, num_bits, A, B)
        
        update_table(generation, best_individual, best_index, best_value, best_fitness)
        
        pairs = select_pairs(population, pop_size)
        offspring = []
        for pair in pairs:
            if random.random() < crossover_rate:
                off1, off2 = crossover(pair, population, num_bits)
                offspring.append(mutate(off1, num_bits, mutation_rate_gene))
                offspring.append(mutate(off2, num_bits, mutation_rate_gene))
        
        population.extend(offspring)
        
        fitnesses = evaluate_population(population, num_bits, A, B)
        population = prune_population(population, fitnesses, min(pop_size, max_population))

        plt.figure(figsize=(10, 6))
        plt.plot(best_fitnesses, label='Mejor aptitud')
        plt.plot(worst_fitnesses, label='Peor aptitud')
        plt.plot(average_fitnesses, label='Aptitud promedio')
        plt.xlabel('Generación')
        plt.ylabel('Aptitud')
        plt.title(f'Evolución de la Aptitud de la Población (Minimización) - Generación {generation}')
        plt.legend()
        plt.savefig(f'gen_images/gen_{generation}.png')
        plt.close()
    
    create_video('gen_images', 'evolution_minimization.mp4')

# Función para actualizar la tabla con el mejor individuo por generación
def update_table(generation, individual, index, value, fitness):
    table.insert("", "end", values=(generation, individual, index, value, fitness))

# Crear un video a partir de las imágenes guardadas
def create_video(image_folder, output_video):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# Interfaz gráfica
def run_gui():
    global table  # Hacer que table sea una variable global
    def start_algorithm():
        try:
            A = float(entry_A.get())
            B = float(entry_B.get())
            delta_x = float(entry_delta_x.get())
            generations = int(entry_generations.get())
            maximize = entry_maximize.get().lower() == 's'
            mutation_rate_gene = float(entry_mutation_rate_gene.get())
            pop_size = int(entry_pop_size.get())
            max_population = int(entry_max_population.get())
            crossover_rate = float(entry_crossover_rate.get())

            # Validaciones
            if A >= B:
                raise ValueError("El valor inicial A no debe ser mayor o igual al valor final B.")
            if not (0 < delta_x <= 1):
                raise ValueError("DeltaX debe estar en el rango de 0 a 1.")
            if pop_size >= max_population:
                raise ValueError("El número de individuos no debe ser mayor o igual al tamaño de la población máxima.")

            # Limpiar imágenes de la carpeta 'gen_images'
            for filename in os.listdir('gen_images'):
                file_path = os.path.join('gen_images', filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

            if maximize:
                genetic_algorithm_maximize(A, B, delta_x, generations, mutation_rate_gene, pop_size, max_population, crossover_rate)
            else:
                genetic_algorithm_minimize(A, B, delta_x, generations, mutation_rate_gene, pop_size, max_population, crossover_rate)

        except ValueError as e:
            print(f"Error en la entrada: {e}")

    root = tk.Tk()
    root.title("Algoritmo Genético")

    tk.Label(root, text="Valor inicial (A):").grid(row=0, column=0, padx=5, pady=5)
    entry_A = tk.Entry(root)
    entry_A.grid(row=0, column=1, padx=5, pady=5)

    tk.Label(root, text="Valor final (B):").grid(row=1, column=0, padx=5, pady=5)
    entry_B = tk.Entry(root)
    entry_B.grid(row=1, column=1, padx=5, pady=5)

    tk.Label(root, text="DeltaX:").grid(row=2, column=0, padx=5, pady=5)
    entry_delta_x = tk.Entry(root)
    entry_delta_x.grid(row=2, column=1, padx=5, pady=5)

    tk.Label(root, text="Número de generaciones:").grid(row=3, column=0, padx=5, pady=5)
    entry_generations = tk.Entry(root)
    entry_generations.grid(row=3, column=1, padx=5, pady=5)

    tk.Label(root, text="Maximizar función (s/n):").grid(row=4, column=0, padx=5, pady=5)
    entry_maximize = tk.Entry(root)
    entry_maximize.grid(row=4, column=1, padx=5, pady=5)

    tk.Label(root, text="Probabilidad de mutación del gen:").grid(row=5, column=0, padx=5, pady=5)
    entry_mutation_rate_gene = tk.Entry(root)
    entry_mutation_rate_gene.grid(row=5, column=1, padx=5, pady=5)

    tk.Label(root, text="Número de individuos:").grid(row=6, column=0, padx=5, pady=5)
    entry_pop_size = tk.Entry(root)
    entry_pop_size.grid(row=6, column=1, padx=5, pady=5)

    tk.Label(root, text="Población máxima:").grid(row=7, column=0, padx=5, pady=5)
    entry_max_population = tk.Entry(root)
    entry_max_population.grid(row=7, column=1, padx=5, pady=5)

    tk.Label(root, text="Tasa de cruza:").grid(row=8, column=0, padx=5, pady=5)
    entry_crossover_rate = tk.Entry(root)
    entry_crossover_rate.grid(row=8, column=1, padx=5, pady=5)

    # Botón para iniciar el algoritmo
    start_button = tk.Button(root, text="Iniciar Algoritmo", command=start_algorithm)
    start_button.grid(row=9, column=0, columnspan=2, pady=10)

    # Tabla para mostrar el mejor individuo por generación
    table_frame = tk.Frame(root)
    table_frame.grid(row=10, column=0, columnspan=2, padx=5, pady=5)

    table = ttk.Treeview(table_frame, columns=("generation", "bits", "index", "x_value", "fitness"), show='headings')
    table.heading("generation", text="Generación")
    table.heading("bits", text="Cadena de Bits")
    table.heading("index", text="Índice")
    table.heading("x_value", text="Valor de X")
    table.heading("fitness", text="Aptitud")
    table.pack()

    root.mainloop()

# Ejecutar la interfaz gráfica
run_gui()
