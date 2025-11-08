import os
import time
import requests
from ddgs import DDGS
from PIL import Image
from io import BytesIO

def descargar_imagenes_duckduckgo(query, carpeta, cantidad=100, delay=1.0):
    os.makedirs(carpeta, exist_ok=True)
    print(f"Descargando imágenes para: '{query}'")

    # Usar la nueva librería ddgs
    with DDGS() as ddgs:
        resultados = ddgs.images(query, max_results=cantidad)

        for i, img in enumerate(resultados, 1):
            try:
                url = img["image"]
                headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0"}
                resp = requests.get(url, headers=headers, timeout=10)

                # Validar respuesta
                if resp.status_code != 200:
                    print(f"Error HTTP {resp.status_code} con {url}")
                    continue

                img_data = resp.content
                image = Image.open(BytesIO(img_data))
                ext = image.format.lower() if image.format else "jpg"
                file_path = os.path.join(carpeta, f"{query.replace(' ', '_')}_{i}.{ext}")

                with open(file_path, "wb") as f:
                    f.write(img_data)
                print(f"[{i}] {file_path}")

                time.sleep(delay)  # Esperar entre descargas para evitar bloqueo

            except Exception as e:
                print(f"Error con imagen {i}: {e}")
                continue


if __name__ == "__main__":
    descargar_imagenes_duckduckgo("external roof", "ceilingGood", 100)
