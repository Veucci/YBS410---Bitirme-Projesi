from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import SegmentationModel
import os
import shutil
from typing import List, Dict
from pydantic import BaseModel
import uuid
from pathlib import Path
import numpy as np
import cv2
import math
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import json

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global model variable
model = None

# Dayanıklılık değerlendirme kriterleri (SIKILI DEĞERLENDİRME)
DAYANIKLILIK_KRITERLERI = {
    "mukemmel": {"min_oran": 0, "max_oran": 2, "puan": 100},
    "cok_iyi": {"min_oran": 2, "max_oran": 5, "puan": 85},
    "iyi": {"min_oran": 5, "max_oran": 10, "puan": 70},
    "orta": {"min_oran": 10, "max_oran": 20, "puan": 55},
    "zayif": {"min_oran": 20, "max_oran": 40, "puan": 40},
    "cok_zayif": {"min_oran": 40, "max_oran": 100, "puan": 25}
}

# Hasar kategorileri
HASAR_KATEGORILERI = {
    0: "Seviye 1 Hasar (Hafif)",
    1: "Seviye 2 Hasar (Orta-Ağır)"
}

# Görsel işleme parametreleri
MAX_TILE_SIZE = 640
OVERLAP_RATIO = 0.1

# Renk paleti hasar seviyelerine göre
HASAR_RENKLERI = {
    0: (255, 255, 0, 100),    # Sarı - Hafif hasar (RGBA)
    1: (255, 0, 0, 120)       # Kırmızı - Ağır hasar (RGBA)
}

class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    mask: List[List[float]]
    area: float

class DayaniklilikAnalizi(BaseModel):
    kategori: str
    puan: int
    aciklama: str
    toplam_hasar_orani: float
    seviye1_orani: float
    seviye2_orani: float

class PredictionResult(BaseModel):
    filename: str
    predictions: List[DetectionResult]
    success: bool
    message: str = ""
    logs: List[str] = []
    image_size: dict = {"width": 0, "height": 0}
    dayaniklilik_analizi: DayaniklilikAnalizi = None
    tile_count: int = 0
    processing_time: float = 0.0
    processed_image_base64: str = None  # Yeni alan

def gorsel_parcalara_bol(image):
    """
    Büyük görseli 640x640 parçalara böler
    """
    h, w = image.shape[:2]
    
    # Eğer görsel zaten küçükse parçalamaya gerek yok
    if h <= MAX_TILE_SIZE and w <= MAX_TILE_SIZE:
        return [(image, 0, 0, 1.0, 1.0)]
    
    parcalar = []
    overlap = int(MAX_TILE_SIZE * OVERLAP_RATIO)
    step_size = MAX_TILE_SIZE - overlap
    
    # Y ekseni boyunca parçala
    for y in range(0, h, step_size):
        y_end = min(y + MAX_TILE_SIZE, h)
        
        # X ekseni boyunca parçala
        for x in range(0, w, step_size):
            x_end = min(x + MAX_TILE_SIZE, w)
            
            # Parçayı kes
            parca = image[y:y_end, x:x_end]
            
            # Parçayı 640x640'a yeniden boyutlandır
            parca_h, parca_w = parca.shape[:2]
            scale_x = MAX_TILE_SIZE / parca_w
            scale_y = MAX_TILE_SIZE / parca_h
            
            parca_resized = cv2.resize(parca, (MAX_TILE_SIZE, MAX_TILE_SIZE))
            
            parcalar.append((parca_resized, x, y, scale_x, scale_y))
    
    return parcalar

def koordinatlari_donustur(detections, x_offset, y_offset, scale_x, scale_y, original_w, original_h):
    """
    Parça koordinatlarını orijinal görsel koordinatlarına dönüştürür
    """
    donusturulmus_tespitler = []
    
    for detection in detections:
        if hasattr(detection, 'masks') and detection.masks is not None:
            # Segmentasyon için mask koordinatlarını dönüştür
            for i, mask in enumerate(detection.masks.data):
                class_id = int(detection.boxes.data[i][5].item())
                confidence = float(detection.boxes.data[i][4].item())
                
                # Mask'ı orijinal boyuta dönüştür
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (int(MAX_TILE_SIZE / scale_x), int(MAX_TILE_SIZE / scale_y)))
                
                # Koordinatları orijinal görsele dönüştür
                mask_original = np.zeros((original_h, original_w), dtype=np.float32)
                
                # Parçanın orijinal görseldeki konumunu hesapla
                y_start = y_offset
                y_end = min(y_offset + mask_resized.shape[0], original_h)
                x_start = x_offset
                x_end = min(x_offset + mask_resized.shape[1], original_w)
                
                # Mask'ı yerleştir
                mask_h = y_end - y_start
                mask_w = x_end - x_start
                mask_original[y_start:y_end, x_start:x_end] = mask_resized[:mask_h, :mask_w]
                
                # Contour'ları çıkar
                mask_binary = (mask_original > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                mask_points = []
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    mask_points = contour.reshape(-1, 2).tolist()
                
                area = np.sum(mask_original > 0.5)
                
                donusturulmus_tespitler.append({
                    'class_id': class_id,
                    'confidence': confidence,
                    'mask': mask_points,
                    'area': area
                })
        
        elif hasattr(detection, 'boxes') and detection.boxes is not None:
            # Detection için bounding box koordinatlarını dönüştür
            for box in detection.boxes.data:
                x1, y1, x2, y2 = box[:4]
                class_id = int(box[5].item())
                confidence = float(box[4].item())
                
                # Koordinatları parça boyutundan orijinal boyuta dönüştür
                x1_orig = (x1 / scale_x) + x_offset
                y1_orig = (y1 / scale_y) + y_offset
                x2_orig = (x2 / scale_x) + x_offset
                y2_orig = (y2 / scale_y) + y_offset
                
                # Sınırları kontrol et
                x1_orig = max(0, min(x1_orig, original_w))
                y1_orig = max(0, min(y1_orig, original_h))
                x2_orig = max(0, min(x2_orig, original_w))
                y2_orig = max(0, min(y2_orig, original_h))
                
                area = (x2_orig - x1_orig) * (y2_orig - y1_orig)
                
                # Bounding box'ı polygon olarak döndür
                mask_points = [
                    [x1_orig, y1_orig],
                    [x2_orig, y1_orig],
                    [x2_orig, y2_orig],
                    [x1_orig, y2_orig]
                ]
                
                donusturulmus_tespitler.append({
                    'class_id': class_id,
                    'confidence': confidence,
                    'mask': mask_points,
                    'area': area
                })
    
    return donusturulmus_tespitler

def tespitleri_birlestir(tum_tespitler):
    """
    Farklı parçalardan gelen tespitleri birleştirir ve örtüşenleri temizler
    """
    if not tum_tespitler:
        return {"tahmin_alanlari": {0: 0, 1: 0}, "tahmin_sayisi": 0, "tespitler": []}
    
    # Sınıf bazında tespitleri grupla
    sinif_bazli_tespitler = {0: [], 1: []}
    
    for tespit in tum_tespitler:
        class_id = tespit['class_id']
        if class_id in sinif_bazli_tespitler:
            sinif_bazli_tespitler[class_id].append(tespit)
    
    # Her sınıf için alanları topla
    tahmin_alanlari = {0: 0, 1: 0}
    toplam_tespit_sayisi = 0
    tum_tespitler_listesi = []
    
    for class_id, tespitler in sinif_bazli_tespitler.items():
        for tespit in tespitler:
            tahmin_alanlari[class_id] += tespit['area']
            toplam_tespit_sayisi += 1
            tum_tespitler_listesi.append(tespit)
    
    return {
        "tahmin_alanlari": tahmin_alanlari,
        "tahmin_sayisi": toplam_tespit_sayisi,
        "tespitler": tum_tespitler_listesi
    }

def dayaniklilik_puani_hesapla(hasar_orani):
    """
    Hasar oranına göre dayanıklılık puanı hesaplar
    """
    for kategori, kriter in DAYANIKLILIK_KRITERLERI.items():
        if kriter["min_oran"] <= hasar_orani < kriter["max_oran"]:
            return {
                "kategori": kategori,
                "puan": kriter["puan"],
                "aciklama": dayaniklilik_aciklamasi(kategori, hasar_orani)
            }
    
    # En yüksek kategoriden büyükse
    return {
        "kategori": "kritik",
        "puan": 10,
        "aciklama": f"KRİTİK HASAR (%{hasar_orani:.1f}) - Acil müdahale gerekli!"
    }

def dayaniklilik_aciklamasi(kategori, hasar_orani):
    """
    Dayanıklılık kategorisi için açıklama döndürür
    """
    aciklamalar = {
        "mukemmel": f"Yapı mükemmel durumda (%{hasar_orani:.1f} hasar), minimal müdahale gerekiyor.",
        "cok_iyi": f"Yapı çok iyi durumda (%{hasar_orani:.1f} hasar), rutin bakım yeterli.",
        "iyi": f"Yapı iyi durumda (%{hasar_orani:.1f} hasar), önleyici bakım önerilir.",
        "orta": f"Yapı orta durumda (%{hasar_orani:.1f} hasar), onarım planlanmalı.",
        "zayif": f"Yapı zayıf durumda (%{hasar_orani:.1f} hasar), acil onarım gerekli.",
        "cok_zayif": f"Yapı çok zayıf durumda (%{hasar_orani:.1f} hasar), kapsamlı müdahale şart."
    }
    return aciklamalar.get(kategori, f"Bilinmeyen durum (%{hasar_orani:.1f} hasar)")

def gorsel_uzerinde_hasarlari_goster(original_image_path, tespitler, image_size):
    """
    Orijinal görsel üzerinde tespit edilen hasarları görselleştirir
    """
    try:
        # Orijinal görseli yükle
        image = cv2.imread(original_image_path)
        if image is None:
            return None
        
        # BGR'dan RGB'ye çevir
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # PIL Image'a çevir
        pil_image = Image.fromarray(image_rgb)
        
        # Şeffaf overlay oluştur
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Font yükle (sistem fontunu kullan)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Her tespit için overlay çiz
        for i, tespit in enumerate(tespitler):
            class_id = tespit['class_id']
            confidence = tespit['confidence']
            mask_points = tespit['mask']
            
            if len(mask_points) >= 3:  # En az 3 nokta gerekli
                # Polygon çiz
                color = HASAR_RENKLERI.get(class_id, (128, 128, 128, 100))
                points = [(int(p[0]), int(p[1])) for p in mask_points]
                draw.polygon(points, fill=color, outline=color[:3] + (255,))
                
                # Güven skorunu yaz
                if points:
                    # Polygon'un merkez noktasını hesapla
                    center_x = sum(p[0] for p in points) // len(points)
                    center_y = sum(p[1] for p in points) // len(points)
                    
                    # Güven skorunu ve sınıf adını yaz
                    class_name = HASAR_KATEGORILERI.get(class_id, f"Class_{class_id}")
                    text = f"{class_name}\n%{confidence*100:.1f}"
                    
                    # Metin arka planı
                    bbox = draw.textbbox((center_x, center_y), text, font=font)
                    draw.rectangle(bbox, fill=(0, 0, 0, 180))
                    draw.text((center_x, center_y), text, fill=(255, 255, 255), font=font, anchor="mm")
        
        # Overlay'ı orijinal görsel ile birleştir
        pil_image = pil_image.convert('RGBA')
        result = Image.alpha_composite(pil_image, overlay)
        
        # RGB'ye çevir
        result = result.convert('RGB')
        
        # Base64'e çevir
        buffer = BytesIO()
        result.save(buffer, format='JPEG', quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_str}"
        
    except Exception as e:
        print(f"Görsel işleme hatası: {str(e)}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model
    # Add SegmentationModel to PyTorch's safe globals before loading model
    torch.serialization.add_safe_globals([SegmentationModel])
    model = YOLO("ai/model.pt")
    print("Model loaded successfully")
    yield
    # Clean up on shutdown
    print("Shutting down application")

app = FastAPI(title="Gelişmiş Görsel İşleme API - Dayanıklılık Analizi", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-image/", response_model=PredictionResult)
async def process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique filename
    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename
    
    start_time = datetime.now()
    
    try:
        # Save uploaded file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Processing image: {file_path}")
        
        # Load image
        image = cv2.imread(str(file_path))
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        original_h, original_w = image.shape[:2]
        image_area = original_h * original_w
        
        logs = []
        logs.append(f"🔍 Görsel yüklendi: {original_w}x{original_h} piksel")
        
        # Büyük görsel işleme: Parçalara böl
        parcalar = gorsel_parcalara_bol(image)
        logs.append(f"🔧 Görsel {len(parcalar)} parçaya bölündü (640x640 tile sistemi)")
        
        tum_tespitler = []
        
        # Her parçayı işle
        for i, (parca, x_offset, y_offset, scale_x, scale_y) in enumerate(parcalar):
            try:
                # Model ile tahmin yap
                results = model.predict(parca, verbose=False, retina_masks=True)
                
                # Koordinatları dönüştür ve tespitleri topla
                for result in results:
                    donusturulmus = koordinatlari_donustur(
                        [result], x_offset, y_offset, scale_x, scale_y, original_w, original_h
                    )
                    tum_tespitler.extend(donusturulmus)
            
            except Exception as e:
                logs.append(f"⚠️ Parça {i+1} işlenirken hata: {str(e)}")
                continue
        
        # Tespitleri birleştir
        birlestirilmis_sonuc = tespitleri_birlestir(tum_tespitler)
        logs.append(f"✅ {len(parcalar)} parça işlendi, {birlestirilmis_sonuc['tahmin_sayisi']} tespit bulundu")
        
        # Dayanıklılık analizi hesapla
        tahmin_alanlari = birlestirilmis_sonuc["tahmin_alanlari"]
        seviye1_alan = tahmin_alanlari.get(0, 0)
        seviye2_alan = tahmin_alanlari.get(1, 0)
        toplam_hasar_alani = seviye1_alan + seviye2_alan
        
        toplam_hasar_orani = (toplam_hasar_alani / image_area) * 100
        seviye1_orani = (seviye1_alan / image_area) * 100
        seviye2_orani = (seviye2_alan / image_area) * 100
        
        dayaniklilik = dayaniklilik_puani_hesapla(toplam_hasar_orani)
        
        logs.append(f"📊 Hasar Analizi:")
        logs.append(f"   • Toplam hasar oranı: %{toplam_hasar_orani:.2f}")
        logs.append(f"   • Seviye 1 hasar: %{seviye1_orani:.2f}")
        logs.append(f"   • Seviye 2 hasar: %{seviye2_orani:.2f}")
        logs.append(f"   • Dayanıklılık kategorisi: {dayaniklilik['kategori']}")
        logs.append(f"   • Dayanıklılık puanı: {dayaniklilik['puan']}/100")
        
        # Sonuçları hazırla
        predictions = []
        for tespit in birlestirilmis_sonuc['tespitler']:
            class_id = tespit['class_id']
            class_name = HASAR_KATEGORILERI.get(class_id, f"Class_{class_id}")
            
            predictions.append(DetectionResult(
                class_id=class_id,
                class_name=class_name,
                confidence=tespit['confidence'],
                mask=tespit['mask'],
                area=tespit['area']
            ))
        
        # İşlem süresini hesapla
        processing_time = (datetime.now() - start_time).total_seconds()
        
        dayaniklilik_analizi = DayaniklilikAnalizi(
            kategori=dayaniklilik['kategori'],
            puan=dayaniklilik['puan'],
            aciklama=dayaniklilik['aciklama'],
            toplam_hasar_orani=round(toplam_hasar_orani, 2),
            seviye1_orani=round(seviye1_orani, 2),
            seviye2_orani=round(seviye2_orani, 2)
        )
        
        # Görsel üzerinde hasarları göster
        processed_image_base64 = gorsel_uzerinde_hasarlari_goster(str(file_path), birlestirilmis_sonuc['tespitler'], {"width": original_w, "height": original_h})
        
        result = PredictionResult(
            filename=unique_filename,
            predictions=predictions,
            success=True,
            logs=logs,
            image_size={"width": original_w, "height": original_h},
            dayaniklilik_analizi=dayaniklilik_analizi,
            tile_count=len(parcalar),
            processing_time=round(processing_time, 2),
            processed_image_base64=processed_image_base64
        )
        
        return result
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup uploaded file
        if file_path.exists():
            file_path.unlink()
            print(f"Cleaned up file: {file_path}")

@app.get("/health")
async def health_check():
    return {"status": "Healthy", "model_loaded": model is not None}

@app.get("/model-info")
async def model_info():
    return {
        "model_type": "YOLO with Advanced Tiling System",
        "classes": list(HASAR_KATEGORILERI.values()),
        "input_format": "image/jpeg, image/png",
        "version": "2.0",
        "features": [
            "640x640 tile processing for large images",
            "Structural durability analysis",
            "Multi-level damage classification",
            "Mathematical damage calculations"
        ],
        "durability_criteria": DAYANIKLILIK_KRITERLERI
    }

@app.get("/durability-criteria")
async def get_durability_criteria():
    """Dayanıklılık değerlendirme kriterlerini döndürür"""
    return {
        "criteria": DAYANIKLILIK_KRITERLERI,
        "damage_categories": HASAR_KATEGORILERI,
        "description": "Sıkılaştırılmış değerlendirme kriterleri kullanılmaktadır"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)