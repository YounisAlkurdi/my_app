# file: enhanced_image_analysis_pro.py
import cv2
import numpy as np
import json
import os
from sklearn.cluster import KMeans
from datetime import datetime
from itertools import combinations

USE_ULTRALYTICS = True
try:
    if USE_ULTRALYTICS:
        from ultralytics import YOLO
        yolo_model = None
except Exception as e:
    print("ultralytics not available:", e)
    USE_ULTRALYTICS = False
    yolo_model = None

# --------------------------
# ImageReader
# --------------------------
class ImageReader:
    def __init__(self, path_or_bytes):
        if isinstance(path_or_bytes, bytes):
            np_img = np.frombuffer(path_or_bytes, np.uint8)
            self.img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        else:
            self.img = cv2.imread(path_or_bytes)
        self.valid = self.img is not None

    def size(self):
        h, w, _ = self.img.shape
        return h, w

    def aspect_ratio(self):
        w, h = self.size()[1], self.size()[0]
        gcd = np.gcd(w, h)
        return f"{w//gcd}:{h//gcd}"

    def resolution(self):
        w, h = self.size()[1], self.size()[0]
        if w >= 3840 and h >= 2160:
            return "4K / UHD"
        elif w >= 2560 and h >= 1440:
            return "2.5K / QHD"
        elif w >= 1920 and h >= 1080:
            return "FHD / 2K"
        elif w >= 1280 and h >= 720:
            return "HD"
        else:
            return "SD"

# --------------------------
# ImageEnhancerSmartQuality
# --------------------------
class ImageEnhancerSmartQuality:
    def enhance(self, image):
        original = image.copy()
        h, w = original.shape[:2]

        lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        lab_merge = cv2.merge((cl, a, b))
        enhanced_colors = cv2.cvtColor(lab_merge, cv2.COLOR_LAB2BGR)

        kernel = np.array([[0, -0.15, 0],
                           [-0.15, 1.6, -0.15],
                           [0, -0.15, 0]])
        sharpened = cv2.filter2D(enhanced_colors, -1, kernel)

        upscale_factor = 1.2
        new_w, new_h = int(w*upscale_factor), int(h*upscale_factor)
        final = cv2.resize(sharpened, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return final

# --------------------------
# Dominant colors
# --------------------------
def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def dominant_colors(roi_bgr, k=3, resize_for_speed=True):
    img = roi_bgr
    if resize_for_speed:
        h, w = img.shape[:2]
        max_dim = 200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1,3).astype(float)
    k = min(k, len(pixels))
    if k <= 0: 
        return [{"hex":"#000000","percent":1.0,"rgb":[0,0,0]}]
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    counts = np.bincount(labels)
    total = counts.sum()
    results = []
    for cnt, center in sorted(zip(counts, centers), key=lambda x: -x[0]):
        percent = float(cnt)/float(total)
        hexc = rgb_to_hex(center)
        results.append({"hex": hexc, "percent": round(percent,3), "rgb":[int(x) for x in center]})
    if not results:
        avg_color = np.mean(pixels, axis=0)
        results = [{"hex": rgb_to_hex(avg_color), "percent":1.0, "rgb":[int(x) for x in avg_color]}]
    return results

# --------------------------
# YOLO loading
# --------------------------
def load_yolo_model(model_name="yolov8x.pt"):
    global yolo_model
    if not USE_ULTRALYTICS:
        raise RuntimeError("ultralytics غير مثبت.")
    if yolo_model is None:
        yolo_model = YOLO(model_name)
    return yolo_model

# --------------------------
# Spatial relations
# --------------------------
def spatial_relation(obj_a, obj_b):
    ax, ay = obj_a['center']['cx'], obj_a['center']['cy']
    bx, by = obj_b['center']['cx'], obj_b['center']['cy']
    horizontal = "يسار" if ax < bx else "يمين" if ax > bx else "في نفس المحور الأفقي"
    vertical = "أعلى" if ay < by else "أسفل" if ay > by else "في نفس المحور الرأسي"
    return f"{obj_a['label']} {horizontal} و {vertical} بالنسبة لـ {obj_b['label']}"

# --------------------------
# Draw objects on image
# --------------------------
def draw_objects(image_bgr, objects):
    img = image_bgr.copy()
    for obj in objects:
        bbox = obj['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        label = obj['label']
        conf = obj['confidence']
        colors = obj.get('dominant_colors', [])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({conf})"
        cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        for i, c in enumerate(colors):
            rgb = tuple(c['rgb'])
            cv2.rectangle(img, (x1+i*20, y2-20), (x1+(i+1)*20, y2), rgb[::-1], -1)
    return img

# --------------------------
# Get object thumbnails
# --------------------------
def get_object_thumbnails(image_bgr, objects, size=(80,80)):
    thumbnails = []
    for obj in objects:
        bbox = obj['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        roi = image_bgr[y1:y2, x1:x2].copy()
        if roi.size==0:
            roi = np.zeros((size[1], size[0],3),dtype=np.uint8)
        thumb = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
        thumbnails.append(thumb)
    return thumbnails

# --------------------------
# Detect and describe
# --------------------------
def detect_and_describe(image_bgr, model=None, conf_threshold=0.3, top_k_colors=3):
    h_img, w_img = image_bgr.shape[:2]
    descriptions = []
    draw_img = image_bgr.copy()

    if model is None:
        raise RuntimeError("لا يوجد موديل كشف مُمرر.")

    results = model.predict(source=image_bgr, imgsz=640, conf=conf_threshold, device='cpu', verbose=False)
    res = results[0]

    boxes = getattr(res, "boxes", None)
    names = model.names if hasattr(model, "names") else None

    if boxes is None or len(boxes)==0:
        return descriptions, draw_img

    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        label = names[cls_id] if names and cls_id in names else str(cls_id)

        x1, y1, x2, y2 = [int(round(x)) for x in xyxy]
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(w_img-1,x2), min(h_img-1,y2)
        w_box, h_box = x2-x1, y2-y1
        cx, cy = x1 + w_box//2, y1 + h_box//2

        roi = image_bgr[y1:y2, x1:x2].copy()
        colors = dominant_colors(roi, k=top_k_colors, resize_for_speed=False) if roi.size>0 else [{"hex":"#000000","percent":1.0,"rgb":[0,0,0]}]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.size>0 else np.zeros((1,1),dtype=np.uint8)
        mean_lum, std_lum = float(np.mean(gray)), float(np.std(gray))

        obj = {
            "id": i,
            "label": label,
            "class_id": cls_id,
            "confidence": round(conf,3),
            "bbox":{"x1":x1,"y1":y1,"x2":x2,"y2":y2,"width":w_box,"height":h_box},
            "center":{"cx":cx,"cy":cy},
            "area": w_box*h_box,
            "aspect_ratio": round(w_box/h_box if h_box>0 else 0,3),
            "mean_luminance": round(mean_lum,2),
            "std_luminance": round(std_lum,2),
            "dominant_colors": colors
        }
        descriptions.append(obj)

    relations = [spatial_relation(a,b) for a,b in combinations(descriptions,2)]
    scene_colors = dominant_colors(cv2.resize(image_bgr,(200,200)), k=3)
    avg_lum = float(np.mean(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)))
    scene_description = f"الصورة بحجم {w_img}x{h_img}، متوسط الإضاءة {avg_lum:.2f}, الألوان الأساسية: {', '.join([c['hex'] for c in scene_colors])}. تحتوي على {len(descriptions)} كائنات. العلاقات المكانية بين الكائنات: {', '.join(relations[:5]) if relations else 'لا توجد علاقات.'}"

    return descriptions, draw_img, scene_description, scene_colors, avg_lum

# --------------------------
# Generate Arabic caption
# --------------------------
def generate_arabic_caption(objects, scene_colors, avg_lum):
    lines = []
    for obj in objects:
        label = obj['label']
        colors = [c['hex'] for c in obj['dominant_colors']]
        pos = obj['center']
        x_desc = "على اليسار" if pos['cx'] < 300 else "على اليمين" if pos['cx'] > 900 else "في الوسط"
        y_desc = "أعلى" if pos['cy'] < 200 else "أسفل" if pos['cy'] > 600 else "في المنتصف"
        color_desc = f" بألوان أساسية {', '.join(colors)}" if colors else ""
        lines.append(f"يوجد {label} {x_desc} و{y_desc}{color_desc}.")
    scene_colors_desc = ', '.join([c['hex'] for c in scene_colors]) if scene_colors else "ألوان مختلفة"
    lines.append(f"متوسط إضاءة الصورة: {avg_lum:.1f}. الألوان الأساسية للصورة: {scene_colors_desc}.")
    lines.append(f"عدد الكائنات المكتشفة: {len(objects)}.")
    return " ".join(lines)

# --------------------------
# Main analyze function
# --------------------------
def analyze_image_file(image_path_or_bytes, output_dir="outputs", model_name="yolov8x.pt", enhance_first=True):
    os.makedirs(output_dir, exist_ok=True)
    reader = ImageReader(image_path_or_bytes)
    if not reader.valid:
        raise FileNotFoundError("الصورة غير موجودة أو غير قابلة للقراءة.")

    original_bgr = reader.img
    image_for_detect = ImageEnhancerSmartQuality().enhance(original_bgr) if enhance_first else original_bgr

    model = load_yolo_model(model_name) if USE_ULTRALYTICS else None
    if model is None:
        raise RuntimeError("لا يوجد موديل كشف صالح.")

    objects, drawn, scene_description, scene_colors, avg_lum = detect_and_describe(image_for_detect, model=model)
    annotated_img = draw_objects(image_for_detect, objects)
    arabic_caption = generate_arabic_caption(objects, scene_colors, avg_lum)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir,f"description_{timestamp}.json")
    with open(json_path,"w",encoding="utf-8") as f:
        json.dump({
            "image_path": str(image_path_or_bytes),
            "image_size":{"height":reader.size()[0],"width":reader.size()[1]},
            "aspect_ratio":reader.aspect_ratio(),
            "resolution":reader.resolution(),
            "enhance_first":enhance_first,
            "scene_description": scene_description,
            "arabic_caption": arabic_caption,
            "objects":objects
        },f,ensure_ascii=False,indent=2)

    out_img_path = os.path.join(output_dir,f"annotated_{timestamp}.jpg")
    cv2.imwrite(out_img_path,annotated_img)

    return {
        "json": json_path,
        "annotated_image": out_img_path,
        "objects_count": len(objects),
        "scene_description": scene_description,
        "arabic_caption": arabic_caption,
        "objects": objects,
        "image_for_detect": image_for_detect
    }
