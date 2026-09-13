import numpy as np
import cv2
import os
from typing import Optional

class DetectorBordesPersonalizado:
    """Clase especializada en la detección de bordes y análisis utilizando la infografía de filtros."""

    @staticmethod
    def procesar_imagen_filtrada(ruta_imagen: str) -> None:
        """
        Carga la imagen 'safarifiltros.jpg', realiza la conversión a escala de grises 
        y aplica el operador Canny para extraer los contornos de la infografía.
        """
        if not os.path.exists(ruta_imagen):
            print(f"[ERROR] No se encontró el archivo en disco: '{ruta_imagen}'.")
            return

        # Carga del tensor de color
        imagen_color = cv2.imread(ruta_imagen)
        if imagen_color is None:
            print(f"[ERROR] No se pudo decodificar el archivo '{ruta_imagen}'.")
            return
        
        # Conversión a escala de grises para el procesamiento de gradientes
        imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)
        print(f"[INFO] Imagen '{ruta_imagen}' cargada exitosamente. Aplicando Canny...")

        # Aplicación del detector de bordes Canny
        canny_bordes = cv2.Canny(imagen_gris, 50, 150)

        # Visualización de los resultados en pantallas independientes
        cv2.imshow("Infografia Original - Safarifiltros", imagen_color)
        cv2.imshow("Deteccion de Bordes (Canny)", canny_bordes)

        print("[INFO] Proceso finalizado. Presiona cualquier tecla para cerrar las ventanas...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ==============================================================================
# BLOQUE DE EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("================================================================")
    print("      SESIÓN 5: ANÁLISIS DE BORDES CON 'safarifiltros.jpg'      ")
    print("================================================================")

    # Nombre exacto de la nueva imagen integrada
    nombre_archivo = 'safarifiltros.jpg'

    # Ejecución del pipeline
    DetectorBordesPersonalizado.procesar_imagen_filtrada(nombre_archivo)
    
    print("================================================================")
