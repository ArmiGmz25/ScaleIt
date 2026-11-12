# ScaleIt

Escala tus imágenes **x4 con IA**, directo en tu computadora y desde el navegador.

ScaleIt usa el modelo [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) para agrandar imágenes reconstruyendo detalle, en lugar de solo estirar pixeles como un resize normal. Todo corre de forma local: tus imágenes no se suben a ningún servidor.

## Características

- **Escalado x4** con Real-ESRGAN (por ejemplo, 1000×750 → 4000×3000).
- **Interfaz web** con Gradio: arrastra la imagen, da clic y descarga el resultado en PNG.
- **Imágenes grandes sin llenar la VRAM:** la imagen se procesa por bloques (_tiles_) con margen, así que no quedan costuras.
- **Detección automática de hardware:** usa la GPU NVIDIA si está disponible y, si no, la CPU.
- **Media precisión (float16)** en GPU cuando el modelo lo soporta: alrededor de 2× más rápido en una RTX 3050.

## Requisitos

- **Python 3.11 o 3.12**
- **Recomendado:** GPU NVIDIA con drivers actualizados. Sin GPU funciona igual, pero mucho más lento.
- Unos 4 GB libres en disco (PyTorch con CUDA ocupa bastante).

## Instalación

### 1. Clona el repositorio

```bash
git clone https://github.com/ArmiGmz25/ScaleIt.git
cd ScaleIt
```

### 2. Crea y activa un entorno virtual

```bash
python -m venv .venv
```

- **Windows (PowerShell):** `.venv\Scripts\activate`
- **Linux / macOS:** `source .venv/bin/activate`

> En Windows, si aparece el error _"running scripts is disabled"_, corre una vez
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a activar.

### 3. Instala PyTorch

Ve a [pytorch.org](https://pytorch.org/get-started/locally/), elige tu sistema, **Pip**, **Python** y tu plataforma de cómputo, y corre el comando que te da. Por ejemplo, para NVIDIA:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> ⚠️ En Windows, `pip install torch` sin `--index-url` instala la versión **solo CPU**.

Para verificar que detecta tu GPU:

```bash
python check_gpu.py
```

### 4. Instala el resto de dependencias

```bash
python -m pip install spandrel gradio
```

### 5. Descarga el modelo

Descarga [`RealESRGAN_x4plus.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) (~64 MB) y guárdalo en la carpeta `modelos/`:

```
ScaleIt/
└── modelos/
    └── RealESRGAN_x4plus.pth
```

En Windows también puedes descargarlo desde PowerShell:

```powershell
mkdir modelos
Invoke-WebRequest -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -OutFile "modelos\RealESRGAN_x4plus.pth"
```

## Uso

### Interfaz web

```bash
python app.py
```

Se abre el navegador en `http://127.0.0.1:7860`. Sube una imagen, da clic en **Escalar** y descarga el resultado.

El control **Tamaño de bloque** define en cuántos pedazos se procesa la imagen. Si te quedas sin memoria de GPU, bájalo (128 o 64).

### Desde la terminal

Para una prueba rápida, coloca una imagen llamada `entrada.jpg` en la raíz del proyecto y corre:

```bash
python motor.py
```

El resultado se guarda como `salida.png`.

## Estructura del proyecto

```
ScaleIt/
├── app.py          # Interfaz web con Gradio
├── motor.py        # Lógica de escalado: carga del modelo y procesamiento por bloques
├── check_gpu.py    # Verifica que PyTorch detecte la GPU
└── modelos/        # Modelos .pth (no se incluyen en el repo)
```

## Cómo funciona

1. La imagen se convierte en un tensor `[1, 3, alto, ancho]` con valores entre 0 y 1.
2. Se divide en bloques (256×256 por default). Cada bloque se envía a la GPU con un margen extra de 16 px para que el modelo tenga contexto en los bordes.
3. El modelo escala cada bloque x4, se recorta el margen y el resultado se pega en su lugar dentro de la imagen final, que se arma en la RAM.
4. El tensor resultante se convierte de nuevo en imagen.

## Créditos y licencias

- **Real-ESRGAN** por Xintao Wang et al. — modelo y pesos bajo [licencia BSD 3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE).
  Paper: [_Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data_](https://arxiv.org/abs/2107.10833).
- **[spandrel](https://github.com/chaiNNer-org/spandrel)** por el equipo de chaiNNer (MIT) — carga de modelos.
- **[Gradio](https://www.gradio.app/)** (Apache 2.0) — interfaz web.
- **[PyTorch](https://pytorch.org/)** — motor de redes neuronales.
