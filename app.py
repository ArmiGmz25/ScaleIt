import time
import gradio as gr
from motor import cargar_modelo, escalar_imagen

print("Cargando el modelo...")
modelo = cargar_modelo("modelos/RealESRGAN_x4plus.pth")
print(f"Modelo listo: {modelo.device} ({modelo.dtype})")


def procesar(imagen, tile):
    """Se ejecuta cada que el usuario da click en 'Escalar'"""
    if imagen is None:
        raise gr.Error("Debe subirse una imagen")

    # Tiempo
    inicio = time.time()
    resultado = escalar_imagen(modelo, imagen, tile=int(tile))
    segundos = time.time() - inicio

    # Imagen
    ancho, alto = imagen.size
    nuevo_ancho, nuevo_alto = resultado.size

    # Información
    informacion = f"{ancho} x {alto} -> {nuevo_ancho} x {nuevo_alto} en {segundos:.1f}s"

    return resultado, informacion


# -- Interfaz
with gr.Blocks(title="ScaleIt") as app:
    gr.Markdown("# ScaleIt \nEscala tus imágenes x4 con IA")

    with gr.Row():
        entrada = gr.Image(type="pil", label="Original")
        salida = gr.Image(type="pil", label="Escalada x4", format="png")

    tile = gr.Slider(
        64, 512, value=256, step=64,
        label="Tamaño del bloque",
        info="Puedes bajarlo para consumir menos recursos",
    )
    boton = gr.Button("Escalar", variant="primary")
    info = gr.Markdown()

    # Conectar el botón con la función
    boton.click(fn=procesar, inputs=[entrada, tile], outputs=[salida, info])

if __name__ == "__main__":
    app.launch(inbrowser=True)
