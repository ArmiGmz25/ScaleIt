import numpy
import torch

# Librería para abrir, convertir y guardar imágenes
from PIL import Image

# para cargar y ejecutar de manera automática modelos
from spandrel import ImageModelDescriptor, ModelLoader


# 1. Cargar el modelo
modelo = ModelLoader().load_from_file("modelos/RealESRGAN_x4plus.pth")
assert isinstance(
    modelo, ImageModelDescriptor), "Esto no es un modelo de imágen"
modelo.cuda().eval()

print(f"Modelo: {modelo.architecture.name} | escala x{modelo.scale}")

# 2. Imagen -> Tensor
img = Image.open("entrada.jpg").convert("RGB")
arr = numpy.array(img).astype(numpy.float32) / 255.0
tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()

# 3. Inferencia
salida = modelo(tensor)

# 4. Tensor -> Imagen
arr_out = salida.squeeze(0).permute(1, 2, 0).cpu().numpy()
img_out = Image.fromarray((arr_out * 255).round().astype(numpy.uint8))
img_out.save("salida.png")

print(f"Listo: {img.size} -> {img_out.size}")
