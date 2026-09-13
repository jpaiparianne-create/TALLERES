import os                       # [Bloque 0: Sistema] - Interfaz Win32 para resolver rutas físicas absolutas.
import cv2                      # [Bloque 0: Visión C++] - Convolución espacial y filtrado digital de imágenes.
import numpy as np              # [Bloque 0: Matemáticas] - Álgebra de matrices y gestión continua de tensores.
import matplotlib.pyplot as plt # [Bloque 0: Renderizado] - Mapeo de tensores a interfaz gráfica de usuario.
from typing import Tuple        # [Bloque 0: Tipado] - Estándar estricto para declaraciones de ingeniería.

print("--- INICIANDO PIPELINE DE CONVOLUCIÓN Y FILTRADO ESPACIAL (SESIÓN 4) ---")

class MotorConvolucion:
    """Motor matemático de filtrado espacial y convolución de kernels (Taller 4)."""
    
    def __init__(self):
        # Definición de Kernels especializados para convolución lineal
        self.kernel_sharpen = np.array([[ 0, -1,  0],
                                        [-1,  5, -1],
                                        [ 0, -1,  0]]) # [Bloque 1: Kernel de Realce] - Filtro pasa-altas para agudizar detalles.

    def aplicar_gaussiano(self, imagen: np.ndarray, ksize: Tuple[int, int] = (5, 5)) -> np.ndarray:
        # Convolución con función de distribución normal bidimensional (Reduce ruido Gaussiano)
        return cv2.GaussianBlur(imagen, ksize, sigmaX=0)

    def aplicar_mediana(self, imagen: np.ndarray, ksize: int = 5) -> np.ndarray:
        # Filtro no lineal basado en estadística de orden (Ideal para eliminar ruido tipo "Sal y Pimienta")
        return cv2.medianBlur(imagen, ksize)

    def aplicar_convolucion_personalizada(self, imagen: np.ndarray) -> np.ndarray:
        # Convolución espacial bidimensional usando el kernel de realce (Sharpen)
        return cv2.filter2D(imagen, ddepth=-1, kernel=self.kernel_sharpen)

    def aplicar_bilateral(self, imagen: np.ndarray) -> np.ndarray:
        # Filtro de preservación de bordes que combina dominio espacial y similitud fotométrica
        return cv2.bilateralFilter(imagen, d=9, sigmaColor=75, sigmaSpace=75)

# ==============================================================================
# AUTO-LOCALIZACIÓN Y LECTURA SEGURA (BLINDAJE CONTRA TILDES EN WINDOWS)
# ==============================================================================
directorio_actual = os.path.dirname(os.path.abspath(__file__)) 
nombre_imagen = os.path.join(directorio_actual, '1. AFICHE CREADO.png') 

if not os.path.exists(nombre_imagen):
    print(f"ERROR CRÍTICO: No se encontró la imagen en:\n{directorio_actual}")
else:
    # [BLINDAJE UTF-8]: Lectura cruda de bytes para prevenir fallos en C++ por caracteres especiales
    stream_bytes = np.fromfile(nombre_imagen, dtype=np.uint8) # [Bloque 4: I/O Seguro] - Lectura física con Python nativo.
    img_bgr = cv2.imdecode(stream_bytes, cv2.IMREAD_COLOR)    # [Bloque 4: Decodificador] - Conversión a tensor BGR.

    if img_bgr is None:
        print("ERROR CRÍTICO: El archivo existe pero OpenCV no pudo decodificar el tensor.")
    else:
        print("1. Tensor original cargado con éxito para filtrado espacial.")

        # Instanciación del motor de convolución
        motor = MotorConvolucion()

        # --- EJECUCIÓN DE FILTROS ESPACIALES ---
        img_gauss = motor.aplicar_gaussiano(img_bgr, ksize=(7, 7))
        img_mediana = motor.aplicar_mediana(img_bgr, ksize=5)
        img_sharpen = motor.aplicar_convolucion_personalizada(img_bgr)
        img_bilateral = motor.aplicar_bilateral(img_bgr)

        # Conversión a escala de grises para mostrar gradiente o análisis de bordes derivativo
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = cv2.convertScaleAbs(sobel_x)

        print("2. Procesamiento de convoluciones completado satisfactoriamente.")

        # ==============================================================================
        # DASHBOARD VISUAL DE FILTRADO ESPACIAL (2x3)
        # ==============================================================================
        plt.figure(figsize=(16, 10))                               # [Bloque 9: Canvas UI] - Cuadrícula de 16x10 pulgadas.
        plt.suptitle("TALLER 4: CONVOLUCIÓN Y FILTRADO ESPACIAL", fontsize=16, fontweight='bold')

        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.title("1. Tensor BGR Original")
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(img_gauss, cv2.COLOR_BGR2RGB))
        plt.title("2. Desenfoque Gaussiano (7x7)")
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(img_mediana, cv2.COLOR_BGR2RGB))
        plt.title("3. Filtro de Mediana (No Lineal)")
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(img_sharpen, cv2.COLOR_BGR2RGB))
        plt.title("4. Convolución Sharpen (Realce)")
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2RGB))
        plt.title("5. Filtro Bilateral (Preserva Bordes)")
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.imshow(sobel_x, cmap='gray')
        plt.title("6. Derivada Espacial (Gradiente Sobel X)")
        plt.axis('off')

        plt.tight_layout()                                         # [Bloque 16: Padding] - Optimización de espacios gráficos.
        plt.show()                                                 # [Bloque 16: Despacho OS] - Renderizado en ventana Win32.
