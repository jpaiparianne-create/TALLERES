import numpy as np
import cv2
import os
from typing import Optional, Tuple

class ProcesadorSesionUno:
    """Clase base inicial para la carga, validación y transformaciones fundamentales de imágenes."""

    @staticmethod
    def validar_y_cargar_imagen(ruta_imagen: str) -> Optional[np.ndarray]:
        """
        Valida defensivamente la existencia del recurso en disco y carga 
        la imagen asegurando el manejo correcto de errores.
        """
        if not os.path.exists(ruta_imagen):
            print(f"[ERROR] El archivo especificado no existe en la ruta: '{ruta_imagen}'.")
            return None

        # Carga de la imagen a color (Tensor BGR)
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"[ERROR] OpenCV no pudo decodificar el archivo '{ruta_imagen}'. Verifique el formato.")
            return None

        print(f"[INFO] Imagen '{ruta_imagen}' cargada exitosamente. Dimensiones (H, W, C): {imagen.shape}")
        return imagen

    @staticmethod
    def aplicar_redimensionamiento(imagen: np.ndarray, escala: float = 0.5) -> np.ndarray:
        """
        Aplica un redimensionamiento proporcional (transformación geométrica básica) 
        sobre el tensor de la imagen.
        """
        nuevo_ancho = int(imagen.shape[1] * escala)
        nuevo_alto = int(imagen.shape[0] * escala)
        
        imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR)
        print(f"[INFO] Imagen redimensionada a: {nuevo_ancho}x{nuevo_alto} píxeles.")
        return imagen_redimensionada


# ==============================================================================
# BLOQUE DE EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("================================================================")
    print("      SESIÓN 1: INTRODUCCIÓN, CARGA Y TRANSFORMACIONES BÁSICAS   ")
    print("================================================================")

    # Nombre de tu imagen oficial de prueba integrada
    nombre_archivo = 'safari.jpg'

    # 1. Carga defensiva de la imagen
    img_original = ProcesadorSesionUno.validar_y_cargar_imagen(nombre_archivo)

    if img_original is not None:
        # 2. Aplicación de transformación geométrica básica
        img_reducida = ProcesadorSesionUno.aplicar_redimensionamiento(img_original, escala=0.5)

        # 3. Visualización en pantalla
        cv2.imshow("Imagen Original - Sesion 1", img_original)
        cv2.imshow("Imagen Redimensionada (50%)", img_reducida)

        print("\n[ESTADO] Presiona cualquier tecla en las ventanas para cerrar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("================================================================")
