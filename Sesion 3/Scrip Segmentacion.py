import numpy as np
import cv2
import os
from typing import Tuple, Optional

class SegmentadorMorfologico:
    """Clase especializada en umbralización (binarización por Otsu) y operaciones morfológicas de limpieza."""

    @staticmethod
    def binarizar_otsu(ruta_imagen: str) -> Tuple[Optional[np.ndarray], float]:
        """
        Carga una imagen en escala de grises y calcula automáticamente 
        el umbral óptimo mediante el Método de Otsu.
        """
        if not os.path.exists(ruta_imagen):
            print(f"[ERROR] No se encontró la imagen en disco: '{ruta_imagen}'.")
            return None, 0.0

        # Carga directa de la imagen en escala de grises (1 canal)
        imagen_gris = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if imagen_gris is None:
            print(f"[ERROR] No se pudo decodificar la imagen '{ruta_imagen}'.")
            return None, 0.0

        # Umbralización automática de Otsu para separar objeto y fondo
        t_otsu, imagen_binaria = cv2.threshold(
            imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print(f"[INFO] Umbral óptimo calculado estadísticamente por Otsu (T): {t_otsu}")
        
        return imagen_binaria, float(t_otsu)

    @staticmethod
    def aplicar_limpieza_morfologica(imagen_binaria: np.ndarray, kernel_size: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica operaciones de Apertura (Eliminación de ruido) 
        y Cierre (Relleno de huecos) usando un Elemento Estructurante.
        """
        # Creación del Elemento Estructurante (Kernel) con NumPy
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        # 1. Operación de Apertura (Erosión seguida de Dilatación) -> Limpia ruido exterior
        apertura = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)

        # 2. Operación de Cierre (Dilatación seguida de Erosión) -> Rellena agujeros internos
        cierre = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel)

        print("[INFO] Operaciones morfológicas aplicadas exitosamente.")
        return apertura, cierre


# ==============================================================================
# BLOQUE DE EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("================================================================")
    print("      SESIÓN 3: SEGMENTACIÓN Y LIMPIEZA MORFOLÓGICA           ")
    print("================================================================")

    # Usamos la imagen 'safari.jpg' en esta carpeta
    nombre_archivo = 'safari.jpg'  

    # Ejecución del pipeline de segmentación
    img_bin, umbral_calculado = SegmentadorMorfologico.binarizar_otsu(nombre_archivo)

    if img_bin is not None:
        # Aplicación de morfología matemática
        img_apertura, img_cierre = SegmentadorMorfologico.aplicar_limpieza_morfologica(img_bin, kernel_size=3)

        # Visualización de resultados en ventanas independientes
        cv2.imshow("1. Imagen Binaria (Otsu)", img_bin)
        cv2.imshow("2. Apertura Morfologica (Limpia Ruido)", img_apertura)
        cv2.imshow("3. Cierre Morfologico (Rellena Huecos)", img_cierre)

        print("\n[ESTADO] Presiona cualquier tecla en las ventanas gráficas para finalizar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("================================================================")
