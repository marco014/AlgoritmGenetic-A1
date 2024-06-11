import random
import numpy as np
import math
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import cv2
import os

# Definir la ecuación como una variable global
ecuacion = "math.log(math.fabs(0.5) + 2 * x) * 3 * math.cos(x)"  # Aquí puedo cambiar mi ecuación

# Clase AlgoritmoGenetico
class AlgoritmoGenetico:
    def __init__(self, ecuacion):
        self.ecuacion = ecuacion

    # Función objetivo
    def aptitud(self, x):
        return eval(self.ecuacion)

    # Inicialización de la población
    def inicializar_poblacion(self, tam_poblacion, num_bits):
        return [''.join(random.choice('01') for _ in range(num_bits)) for _ in range(tam_poblacion)]

    # Decodificación de un individuo
    def conv_binario(self, individuo, num_bits, A, B):
        valor_maximo = 2**num_bits - 1
        entero = int(individuo, 2)
        return A + (entero / valor_maximo) * (B - A)

    # Evaluación de la población
    def evaluar_poblacion(self, poblacion, num_bits, A, B):
        return [self.aptitud(self.conv_binario(ind, num_bits, A, B)) for ind in poblacion]

    # Selección de parejas (Estrategia A1)
    def seleccionar_parejas(self, poblacion, n):
        parejas = []
        for i in range(len(poblacion)):
            m = random.randint(0, n)  # Genera un número aleatorio entre 0 y n
            companeros = random.sample([j for j in range(len(poblacion)) if j != i], m)
            # Genera m números aleatorios que hacen referencia a individuos distintos de sí mismo
            parejas.extend([(i, companero) for companero in companeros])
        return parejas

    # Cruza (Estrategia C1)
    def cruza(self, pareja, poblacion, num_bits):
        p1, p2 = pareja
        punto = random.randint(1, num_bits - 1)
        hijo1 = poblacion[p1][:punto] + poblacion[p2][punto:]
        hijo2 = poblacion[p2][:punto] + poblacion[p1][punto:]
        return hijo1, hijo2

    # Mutación (Estrategia M2)
    def mutacion(self, individuo, num_bits, tasa_mutacion_individuo, tasa_mutacion_gen):
        if random.random() < tasa_mutacion_individuo:
            individuo = list(individuo)
            for i in range(num_bits):
                if random.random() < tasa_mutacion_gen:
                    j = random.randint(0, num_bits - 1)
                    # Intercambio de posición de bits
                    individuo[i], individuo[j] = individuo[j], individuo[i]
            individuo = ''.join(individuo)
        return individuo

    # Poda (Estrategia P2)
    def podar_poblacion(self, poblacion, aptitudes, tam_poblacion):
        mejor_individuo = poblacion[np.argmax(aptitudes)]
        indices = list(range(len(poblacion)))
        indices.remove(np.argmax(aptitudes))
        mantener = random.sample(indices, tam_poblacion - 1)
        nueva_poblacion = [poblacion[i] for i in mantener]
        nueva_poblacion.append(mejor_individuo)
        return nueva_poblacion

    # Algoritmo Genético para maximizar la aptitud
    def maximizar_aptitud(self, A, B, delta_x, generaciones, tam_poblacion, max_poblacion):
        num_bits = math.ceil(math.log2((B - A) / delta_x + 1))

        poblacion = self.inicializar_poblacion(tam_poblacion, num_bits)

        mejores_aptitudes = []
        peores_aptitudes = []
        aptitudes_promedio = []

        if not os.path.exists('gen_images'):
            os.makedirs('gen_images')

        for generacion in range(generaciones):
            aptitudes = self.evaluar_poblacion(poblacion, num_bits, A, B)

            mejor_aptitud = max(aptitudes)
            peor_aptitud = min(aptitudes)
            aptitud_promedio = sum(aptitudes) / len(aptitudes)

            mejores_aptitudes.append(mejor_aptitud)
            peores_aptitudes.append(peor_aptitud)
            aptitudes_promedio.append(aptitud_promedio)

            mejor_indice = np.argmax(aptitudes)
            mejor_individuo = poblacion[mejor_indice]
            mejor_valor = self.conv_binario(mejor_individuo, num_bits, A, B)

            ActualizadorTabla.actualizar_tabla(generacion, mejor_individuo, mejor_indice, mejor_valor, mejor_aptitud)

            n = random.randint(1, len(poblacion) - 1)
            tasa_cruce = random.uniform(0.5, 1.0)
            tasa_mutacion_individuo = random.uniform(0.01, 0.1)
            tasa_mutacion_gen = random.uniform(0.01, 0.1)

            parejas = self.seleccionar_parejas(poblacion, n)
            descendientes = []
            for pareja in parejas:
                if random.random() < tasa_cruce:
                    desc1, desc2 = self.cruza(pareja, poblacion, num_bits)
                    descendientes.append(self.mutacion(desc1, num_bits, tasa_mutacion_individuo, tasa_mutacion_gen))
                    descendientes.append(self.mutacion(desc2, num_bits, tasa_mutacion_individuo, tasa_mutacion_gen))

            poblacion.extend(descendientes)

            aptitudes = self.evaluar_poblacion(poblacion, num_bits, A, B)
            poblacion = self.podar_poblacion(poblacion, aptitudes, min(tam_poblacion, max_poblacion))

            GraficoEvolucion.generar_grafico(mejores_aptitudes, peores_aptitudes, aptitudes_promedio, generacion, 'Maximización')

        VideoCreador.crear_video('gen_images', 'evolucion_maximizacion.mp4')

    # Algoritmo Genético para minimizar la aptitud
    def minimizar_aptitud(self, A, B, delta_x, generaciones, tam_poblacion, max_poblacion):
        num_bits = math.ceil(math.log2((B - A) / delta_x + 1))

        poblacion = self.inicializar_poblacion(tam_poblacion, num_bits)

        mejores_aptitudes = []
        peores_aptitudes = []
        aptitudes_promedio = []

        if not os.path.exists('gen_images'):
            os.makedirs('gen_images')

        for generacion in range(generaciones):
            aptitudes = self.evaluar_poblacion(poblacion, num_bits, A, B)

            mejor_aptitud = min(aptitudes)
            peor_aptitud = max(aptitudes)
            aptitud_promedio = sum(aptitudes) / len(aptitudes)

            mejores_aptitudes.append(mejor_aptitud)
            peores_aptitudes.append(peor_aptitud)
            aptitudes_promedio.append(aptitud_promedio)

            mejor_indice = np.argmin(aptitudes)
            mejor_individuo = poblacion[mejor_indice]
            mejor_valor = self.conv_binario(mejor_individuo, num_bits, A, B)

            ActualizadorTabla.actualizar_tabla(generacion, mejor_individuo, mejor_indice, mejor_valor, mejor_aptitud)

            n = random.randint(1, len(poblacion) - 1)
            tasa_cruce = random.uniform(0.5, 1.0)
            tasa_mutacion_individuo = random.uniform(0.01, 0.1)
            tasa_mutacion_gen = random.uniform(0.01, 0.1)

            parejas = self.seleccionar_parejas(poblacion, n)
            descendientes = []
            for pareja in parejas:
                if random.random() < tasa_cruce:
                    desc1, desc2 = self.cruza(pareja, poblacion, num_bits)
                    descendientes.append(self.mutacion(desc1, num_bits, tasa_mutacion_individuo, tasa_mutacion_gen))
                    descendientes.append(self.mutacion(desc2, num_bits, tasa_mutacion_individuo, tasa_mutacion_gen))

            poblacion.extend(descendientes)

            aptitudes = self.evaluar_poblacion(poblacion, num_bits, A, B)
            poblacion = self.podar_poblacion(poblacion, aptitudes, min(tam_poblacion, max_poblacion))

            GraficoEvolucion.generar_grafico(mejores_aptitudes, peores_aptitudes, aptitudes_promedio, generacion, 'Minimización')

        VideoCreador.crear_video('gen_images', 'evolucion_minimizacion.mp4')

# Clase para actualizar la tabla
class ActualizadorTabla:
    @staticmethod
    def actualizar_tabla(generacion, individuo, indice, valor, aptitud):
        tabla.insert("", "end", values=(generacion, individuo, indice, valor, aptitud))

# Clase para generar gráficos de evolución
class GraficoEvolucion:
    @staticmethod
    def generar_grafico(mejores_aptitudes, peores_aptitudes, aptitudes_promedio, generacion, tipo):
        plt.figure(figsize=(10, 6))
        plt.plot(mejores_aptitudes, label='Mejor aptitud')
        plt.plot(peores_aptitudes, label='Peor aptitud')
        plt.plot(aptitudes_promedio, label='Aptitud promedio')
        plt.xlabel('Generación')
        plt.ylabel('Aptitud')
        plt.title(f'Evolución de la Aptitud de la Población ({tipo}) - Generación {generacion}')
        plt.legend()
        plt.savefig(f'gen_images/gen_{generacion}.png')
        plt.close()

# Clase para crear videos
class VideoCreador:
    @staticmethod
    def crear_video(carpeta_imagenes, video_salida):
        imagenes = [img for img in os.listdir(carpeta_imagenes) if img.endswith(".png")]
        imagenes.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        frame = cv2.imread(os.path.join(carpeta_imagenes, imagenes[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_salida, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

        for imagen in imagenes:
            video.write(cv2.imread(os.path.join(carpeta_imagenes, imagen)))

        cv2.destroyAllWindows()
        video.release()

# Interfaz gráfica
def iniciar_gui():
    global tabla

    def iniciar_algoritmo():
        try:
            A = float(entrada_A.get())
            B = float(entrada_B.get())
            delta_x = float(entrada_delta_x.get())
            generaciones = int(entrada_generaciones.get())
            maximizar = entrada_maximizar.get().lower() == 's'
            tam_poblacion = int(entrada_tam_poblacion.get())
            max_poblacion = int(entrada_max_poblacion.get())

            tabla.delete(*tabla.get_children())

            ag = AlgoritmoGenetico(ecuacion)
            if maximizar:
                ag.maximizar_aptitud(A, B, delta_x, generaciones, tam_poblacion, max_poblacion)
            else:
                ag.minimizar_aptitud(A, B, delta_x, generaciones, tam_poblacion, max_poblacion)

        except ValueError:
            print("Por favor, ingrese valores válidos.")

    root = tk.Tk()
    root.title("Algoritmo Genético")

    tk.Label(root, text="Valor de Inicial:").grid(row=0, column=0)
    entrada_A = tk.Entry(root)
    entrada_A.grid(row=0, column=1)

    tk.Label(root, text="Valor de Final:").grid(row=1, column=0)
    entrada_B = tk.Entry(root)
    entrada_B.grid(row=1, column=1)

    tk.Label(root, text="Valor de delta x:").grid(row=2, column=0)
    entrada_delta_x = tk.Entry(root)
    entrada_delta_x.grid(row=2, column=1)

    tk.Label(root, text="Número de generaciones:").grid(row=3, column=0)
    entrada_generaciones = tk.Entry(root)
    entrada_generaciones.grid(row=3, column=1)

    tk.Label(root, text="Maximizar? (s/n):").grid(row=4, column=0)
    entrada_maximizar = tk.Entry(root)
    entrada_maximizar.grid(row=4, column=1)

    tk.Label(root, text="Tamaño de la población:").grid(row=7, column=0)
    entrada_tam_poblacion = tk.Entry(root)
    entrada_tam_poblacion.grid(row=7, column=1)

    tk.Label(root, text="Tamaño máximo de la población:").grid(row=8, column=0)
    entrada_max_poblacion = tk.Entry(root)
    entrada_max_poblacion.grid(row=8, column=1)

    tk.Button(root, text="Ejecutar Algoritmo", command=iniciar_algoritmo).grid(row=11, column=0, columnspan=2)

    columnas = ("Generación", "Individuo", "Índice", "Valor", "Aptitud")
    tabla = ttk.Treeview(root, columns=columnas, show="headings")
    for col in columnas:
        tabla.heading(col, text=col)
    tabla.grid(row=12, column=0, columnspan=2)

    root.mainloop()

# Ejecutar la interfaz gráfica
if __name__ == "__main__":
    iniciar_gui()
