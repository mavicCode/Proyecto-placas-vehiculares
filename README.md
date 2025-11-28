# 🚗 Sistema de Detección y OCR de Placas Vehiculares

Sistema web completo para detectar placas vehiculares en imágenes y extraer su texto mediante OCR, utilizando YOLO11 y EasyOCR.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![YOLO](https://img.shields.io/badge/YOLO-11-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Características

- ✅ **Detección automática** de placas vehiculares usando YOLO11
- ✅ **OCR avanzado** con 6 técnicas de preprocesamiento de imagen
- ✅ **Interfaz web** intuitiva con Flask
- ✅ **Visualización** de todas las etapas de procesamiento
- ✅ **Confianza individual** para cada técnica de preprocesamiento
- ✅ **Resultados en tiempo real** con visualización de bounding boxes

## 🛠️ Tecnologías

- **Backend**: Flask 3.0
- **Detección**: Ultralytics YOLO11
- **OCR**: EasyOCR 1.7.2
- **Procesamiento**: OpenCV, NumPy
- **Frontend**: HTML5, CSS3, JavaScript

## 📋 Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Modelo YOLO entrenado (`best.pt`)

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/proyecto_placas.git
cd proyecto_placas
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Obtener el modelo YOLO

Tienes dos opciones:

**Opción A: Entrenar tu propio modelo**
- Sigue el notebook incluido para entrenar desde cero
- Coloca el archivo `best.pt` en la carpeta `models/`

**Opción B: Descargar modelo pre-entrenado**
- Descarga el modelo desde [enlace] (si tienes uno disponible)
- Coloca el archivo en `models/best.pt`

## 🎯 Uso

### Iniciar el servidor

```bash
python app.py
```

El servidor estará disponible en:
- Local: http://127.0.0.1:5000
- Red: http://[tu-ip]:5000

### Usar la aplicación

1. Abre tu navegador en `http://127.0.0.1:5000`
2. Selecciona una imagen con vehículos
3. Ajusta el umbral de confianza (opcional)
4. Haz clic en "Detectar Placas"
5. Visualiza los resultados con:
   - Placas detectadas
   - Texto extraído
   - Confianza de detección y OCR
   - Pasos de preprocesamiento

## 📁 Estructura del Proyecto

```
proyecto_placas/
├── app.py                 # Aplicación Flask principal
├── requirements.txt       # Dependencias del proyecto
├── README.md             # Este archivo
├── .gitignore            # Archivos ignorados por Git
├── models/               # Modelos YOLO
│   ├── .gitkeep
│   └── best.pt          # Modelo entrenado (no incluido)
├── templates/            # Plantillas HTML
│   └── index.html       # Interfaz principal
├── uploads/              # Imágenes subidas (temporal)
│   └── .gitkeep
└── results/              # Imágenes procesadas (temporal)
    └── .gitkeep
```

## 🔧 Configuración

Puedes ajustar la configuración en `app.py`:

```python
# Configuración básica
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODEL_PATH = 'models/best.pt'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
```

## 🎨 Técnicas de Preprocesamiento

El sistema aplica 6 técnicas avanzadas de procesamiento de imagen:

1. **Original** - Denoising básico
2. **CLAHE** - Mejora de contraste adaptativo
3. **Gamma Correction** - Corrección de iluminación
4. **Unsharp Masking** - Mejora de nitidez
5. **Binary Otsu** - Binarización automática
6. **Binary Adaptive** - Binarización adaptativa

## 📊 Tipos de Imágenes Soportadas

- **Formatos**: PNG, JPG, JPEG
- **Tamaño máximo**: 16 MB
- **Resolución recomendada**: 1280x720 o superior
- **Contenido**: Vehículos con placas visibles

## 🐛 Solución de Problemas

### Error: "Can't get attribute 'C3k2'"
```bash
pip install --upgrade ultralytics
```

### Error: PyTorch weights_only
El código ya incluye la solución automática para PyTorch 2.6+

### El modelo no detecta placas
- Verifica que `models/best.pt` existe
- Asegúrate de que el modelo fue entrenado para placas
- Ajusta el umbral de confianza

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/NuevaCaracteristica`)
3. Commit tus cambios (`git commit -m 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/NuevaCaracteristica`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👥 Autores

- Tu Nombre - [@tu-usuario](https://github.com/tu-usuario)

## 🙏 Agradecimientos

- Dataset: [Car Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) de Kaggle
- Ultralytics por YOLO
- EasyOCR por el motor de OCR
- Comunidad de OpenCV

## 📧 Contacto

Para preguntas o sugerencias:
- Email: tu-email@ejemplo.com
- GitHub: [@tu-usuario](https://github.com/tu-usuario)

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub
