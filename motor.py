import numpy
import torch
import time
# Librería para abrir, convertir y guardar imágenes
from PIL import Image
# para cargar y ejecutar de manera automática modelos
from spandrel import ImageModelDescriptor, ModelLoader


def cargar_modelo(ruta: str) -> ImageModelDescriptor:
    """Carga un modelo de escalado y lo repara en el mejor dispositivo posible"""
    modelo = ModelLoader().load_from_file(ruta)
    if not isinstance(modelo, ImageModelDescriptor):
        raise ValueError(f"{ruta} no es un modelo de imagen a imagen")

    # Usa la GPU si hay, si no hay se usa la CPU
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo.to(dispositivo).eval()

    # Media precisión solo si hay GPU y el modelo lo soporta
    if dispositivo == "cuda" and modelo.supports_half:
        modelo.half()

    return modelo


def escalar_por_bloques(
    modelo: ImageModelDescriptor,
    tensor: torch.Tensor,
    tile: int = 256,
    pad: int = 16,
) -> torch.Tensor:
    """Escala un tensor [1, C, H, W] por bloques para no llenar la VRAM"""
    _, c, h, w = tensor.shape
    s = modelo.scale

    # La imagen final se arma en la CPU y no en la GPU
    salida = torch.zeros((1, c, h * s, w * s))

    total = ((h + tile - 1) // tile) * ((w + tile - 1) // tile)
    n = 0

    for y in range(0, h, tile):
        for x in range(0, w, tile):

            # Coordenadas del bloque sin salirse de la imágen
            y2, x2 = min(y + tile, h), min(x + tile, w)

            # Mismo bloque, pero con margen extra para dar contexto
            py1, px1, = max(y - pad, 0), max(x - pad, 0)
            py2, px2 = min(y2 + pad, h), min(x2 + pad, w)

            # Esta parte es la única que viaja a la GPU
            bloque = tensor[:, :, py1:py2, px1:px2].to(
                modelo.device, modelo.dtype)
            out = modelo(bloque)

            # Recortar el margen del resultado (ya escalado x s)
            oy1, ox1 = (y - py1) * s, (x - px1) * s
            oy2, ox2 = oy1 + (y2 - y) * s, ox1 + (x2 - x) * s

            # Pegar en su lugar dentro de la imagen final
            salida[:, :, y * s:y2 * s, x * s:x2 *
                   s] = out[:, :, oy1:oy2, ox1:ox2].float().cpu()

            n += 1
            print(f"\rBloque {n}/{total}", end="")
    print()
    return salida


def escalar_imagen(
    modelo: ImageModelDescriptor,
    img: Image.Image,
    tile: int = 256,
) -> Image.Image:
    """Recibe una imágen de PIL y regresa la versión escalada"""
    # Imagen -> tensor
    img = img.convert("RGB")
    arr = numpy.array(img).astype(numpy.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    # Escalar
    salida = escalar_por_bloques(modelo, tensor, tile=tile)

    # Tensor -> imagen
    arr_out = salida.squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray((arr_out * 255).round().astype(numpy.uint8))


if __name__ == "__main__":
    import time

    modelo = cargar_modelo("modelos/RealESRGAN_x4plus.pth")
    print(
        f"Modelo: {modelo.architecture.name} | x{modelo.scale} | {modelo.device} | {modelo.dtype}")

    img = Image.open("entrada.jpg")
    inicio = time.time()
    resultado = escalar_imagen(modelo, img)
    print(f"tiempo: {time.time() - inicio:.1f} s")

    resultado.save("salida.png")
    print(f"Listo: {img.size} -> {resultado.size}")
