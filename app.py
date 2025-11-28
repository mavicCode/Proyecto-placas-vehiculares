# app.py - Aplicación Flask principal
"""
Sistema web de detección y OCR de placas vehiculares
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import base64
from datetime import datetime
import json

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/best.pt'  # Ajusta según tu modelo

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Crear carpetas necesarias
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Variables globales para modelos (se cargan una vez)
yolo_model = None
ocr_reader = None

def init_models():
    """Inicializa los modelos YOLO y OCR"""
    global yolo_model, ocr_reader
    
    if yolo_model is None:
        try:
            # Verificar si el archivo existe
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
            
            # Solución para PyTorch 2.6: configurar weights_only=False para modelos confiables
            import torch
            # Guardar configuración original
            original_load = torch.load
            # Redefinir torch.load para usar weights_only=False por defecto
            torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
            
            yolo_model = YOLO(MODEL_PATH)
            print("✓ Modelo YOLO personalizado cargado correctamente")
            
            # Restaurar torch.load original
            torch.load = original_load
        except Exception as e:
            print(f"✗ Error cargando modelo: {e}")
            yolo_model = None
    
    if ocr_reader is None:
        try:
            ocr_reader = easyocr.Reader(['en'], gpu=False)
            print("✓ OCR inicializado")
        except Exception as e:
            print(f"✗ Error inicializando OCR: {e}")

def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_plate(image):
    """
    Preprocesamiento con 6 técnicas clave optimizadas para OCR
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Redimensionar a mayor resolución
    height, width = gray.shape
    target_height = 150
    if height < target_height:
        scale_factor = target_height / height
        new_width = int(width * scale_factor)
        gray = cv2.resize(gray, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    processed_versions = {}
    
    # === TÉCNICA 1: Original (con denoising básico) ===
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    processed_versions['original'] = denoised
    
    # === TÉCNICA 2: CLAHE (Mejora de contraste) ===
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    processed_versions['clahe'] = enhanced
    
    # === TÉCNICA 3: Gamma Correction (Mejora de iluminación) ===
    def apply_gamma(img, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)
    
    gamma_corrected = apply_gamma(enhanced, 1.2)
    processed_versions['gamma'] = gamma_corrected
    
    # === TÉCNICA 4: Unsharp Masking (Mejora de nitidez) ===
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    unsharp = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    processed_versions['unsharp'] = unsharp
    
    # === TÉCNICA 5: Binary Otsu (Binarización automática) ===
    _, binary_otsu = cv2.threshold(gamma_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_versions['binary'] = binary_otsu
    
    # === TÉCNICA 6: Binary Adaptive (Binarización adaptativa) ===
    binary_adaptive = cv2.adaptiveThreshold(
        enhanced, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        15, 5
    )
    processed_versions['adaptive'] = binary_adaptive
    
    return processed_versions

def extract_text_single_version(img_version):
    """Extrae texto de una sola versión de imagen"""
    try:
        results = ocr_reader.readtext(
            img_version, 
            detail=1,
            paragraph=False,
            min_size=8,
            text_threshold=0.5,
            low_text=0.2,
            link_threshold=0.2,
            canvas_size=3000,
            mag_ratio=1.8
        )
        
        if results:
            texts = [text for (bbox, text, conf) in results]
            confidences = [conf for (bbox, text, conf) in results]
            
            full_text = ''.join(texts)
            avg_confidence = np.mean(confidences)
            
            # Limpiar texto
            clean_text = ''.join(c for c in full_text if c.isalnum() or c == '-')
            clean_text = clean_text.strip().upper()
            
            return clean_text, float(avg_confidence)
    except:
        pass
    
    return "NO DETECTADO", 0.0

def extract_text_from_plate(plate_image):
    """
    Extrae texto y genera confianza individual para cada técnica
    """
    processed_versions = preprocess_plate(plate_image)
    
    # Calcular OCR para cada versión
    ocr_results = {}
    best_result = None
    best_confidence = 0
    best_version_name = None
    
    for version_name, img_version in processed_versions.items():
        text, confidence = extract_text_single_version(img_version)
        ocr_results[version_name] = {
            'text': text,
            'confidence': confidence
        }
        
        # Encontrar la mejor
        if confidence > best_confidence and text != "NO DETECTADO":
            best_confidence = confidence
            best_result = text
            best_version_name = version_name
    
    # Si no se detectó nada, buscar cualquier resultado
    if not best_result:
        for version_name, result in ocr_results.items():
            if result['text'] != "NO DETECTADO":
                best_result = result['text']
                best_confidence = result['confidence']
                best_version_name = version_name
                break
    
    if not best_result:
        best_result = "NO DETECTADO"
        best_confidence = 0.0
        best_version_name = 'original'
    
    return (
        best_result,
        best_confidence,
        best_version_name,
        processed_versions,
        ocr_results  # Nuevo: resultados individuales
    )

def process_image(image_path, conf_threshold=0.25):
    """Procesa una imagen y detecta placas con OCR"""
    results = yolo_model.predict(source=image_path, conf=conf_threshold)
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    detected_plates = []
    
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        # Recortar placa con margen
        margin = 5
        y1_crop = max(0, y1 - margin)
        y2_crop = min(img_rgb.shape[0], y2 + margin)
        x1_crop = max(0, x1 - margin)
        x2_crop = min(img_rgb.shape[1], x2 + margin)
        
        crop = img_rgb[y1_crop:y2_crop, x1_crop:x2_crop]
        
        # OCR con información del procesamiento
        plate_text, ocr_confidence, best_version, processed_versions, ocr_results = extract_text_from_plate(crop)
        
        # Convertir imágenes procesadas a base64 para enviar al frontend
        processing_steps = {}
        for version_name, img_data in processed_versions.items():
            # Convertir a base64
            _, buffer = cv2.imencode('.png', img_data)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            processing_steps[version_name] = {
                'image': f"data:image/png;base64,{img_base64}",
                'text': ocr_results[version_name]['text'],
                'confidence': ocr_results[version_name]['confidence']
            }
        
        # Convertir crop original a base64
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', crop_bgr)
        crop_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Dibujar en imagen original
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f'{plate_text} ({conf:.2f})'
        cv2.putText(img_rgb, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Guardar información
        detected_plates.append({
            'id': i + 1,
            'text': plate_text,
            'detection_conf': float(conf),
            'ocr_conf': float(ocr_confidence),
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'best_version': best_version,
            'crop_original': f"data:image/png;base64,{crop_base64}",
            'processing_steps': processing_steps
        })
    
    # Guardar imagen procesada
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_filename = f'result_{timestamp}.jpg'
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    cv2.imwrite(result_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    
    return detected_plates, result_filename

# ============================================================================
# RUTAS DE LA APLICACIÓN
# ============================================================================

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint para subir y procesar imagen"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se envió ningún archivo'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Formato de archivo no permitido'}), 400
        
        # Guardar archivo
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Obtener umbral de confianza
        conf_threshold = float(request.form.get('confidence', 0.25))
        
        # Procesar imagen
        plates, result_filename = process_image(filepath, conf_threshold)
        
        return jsonify({
            'success': True,
            'plates': plates,
            'result_image': f'/results/{result_filename}',
            'total_plates': len(plates)
        })
    
    except Exception as e:
        import traceback
        print(f"✗ Error procesando imagen: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def get_result(filename):
    """Sirve las imágenes procesadas"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/health')
def health_check():
    """Verifica el estado de la aplicación"""
    models_loaded = yolo_model is not None and ocr_reader is not None
    return jsonify({
        'status': 'ok' if models_loaded else 'initializing',
        'models_loaded': models_loaded
    })

# ============================================================================
# INICIALIZACIÓN
# ============================================================================

if __name__ == '__main__':
    print("Inicializando modelos...")
    init_models()
    print("Servidor listo!")
    app.run(debug=True, host='0.0.0.0', port=5000)