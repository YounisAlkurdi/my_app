import cv2
import numpy as np

# ===========================
# كلاس تحليل الصورة
# ===========================
class ImageReader:
    """
    كلاس لتحليل مواصفات الصورة:
    - الأبعاد
    - النسبة البعدية
    - الدقة
    يدعم رفع ملفات bytes أو مسار صورة
    """
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


# ===========================
# كلاس تحسين الصورة الذكي
# ===========================
class ImageEnhancerSmartQuality:
    """
    كلاس لتحسين جودة الصورة:
    - Edge-preserving + DetailEnhance
    - CLAHE لتحسين الألوان
    - Gamma خفيف
    - Sharpen ذكي قليل
    - Upscale خفيف إذا الصورة صغيرة
    """
    def __init__(self):
        pass

    def enhance(self, image):
        original = image.copy()
        h, w = original.shape[:2]

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        mean_lum = np.mean(gray)
        std_lum = np.std(gray)
        is_gray = std_lum < 10

        # قرار على نوع الفلتر
        if is_gray:
            sigma_s, sigma_r = 20, 0.2
            apply_detail = False
        elif mean_lum < 50:
            sigma_s, sigma_r = 30, 0.25
            apply_detail = True
        elif std_lum < 30:
            sigma_s, sigma_r = 40, 0.25
            apply_detail = True
        else:
            sigma_s, sigma_r = 30, 0.2
            apply_detail = False

        # Edge-preserving filter
        smart = cv2.edgePreservingFilter(original, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)

        if apply_detail:
            smart = cv2.detailEnhance(smart, sigma_s=5, sigma_r=0.05)

        # CLAHE لتحسين الألوان
        lab = cv2.cvtColor(smart, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        cl = clahe.apply(l)
        lab_merge = cv2.merge((cl, a, b))
        enhanced_colors = cv2.cvtColor(lab_merge, cv2.COLOR_LAB2BGR)

        # Gamma خفيف
        gamma = 1.02
        lookUpTable = np.array([((i / 255.0)**(1.0/gamma))*255 for i in range(256)]).astype("uint8")
        gamma_corrected = cv2.LUT(enhanced_colors, lookUpTable)

        # Sharpen خفيف
        kernel = np.array([[0, -0.15, 0],
                           [-0.15, 1.5, -0.15],
                           [0, -0.15, 0]])
        sharpened = cv2.filter2D(gamma_corrected, -1, kernel)

        # Upscale ×1.2
        upscale_factor = 1.2
        new_w, new_h = int(w*upscale_factor), int(h*upscale_factor)
        final = cv2.resize(sharpened, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return cv2.cvtColor(final, cv2.COLOR_BGR2RGB)


# -----------------------------
# تجربة كلاسين مباشرة
# -----------------------------
if __name__ == "__main__":
    image_path = "test.jpg"

    # تحليل الصورة
    reader = ImageReader(image_path)
    if reader.valid:
        h, w = reader.size()
        print(f"أبعاد الصورة: {h}x{w}")
        print(f"النسبة البعدية: {reader.aspect_ratio()}")
        print(f"دقة الصورة: {reader.resolution()}")

        # تحسين الصورة
        enhancer = ImageEnhancerSmartQuality()
        enhanced_img = enhancer.enhance(reader.img)

        cv2.imshow("Original", reader.img)
        cv2.imshow("Enhanced Quality", cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(" لم يتم العثور على الصورة")
