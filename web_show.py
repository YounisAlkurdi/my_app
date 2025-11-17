import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# استدعاء كلاسات المعالجة والتحليل
from processors import ImageEnhancerSmartQuality, ImageReader

# -----------------------------
# إعداد الصفحة
# -----------------------------
st.set_page_config(
    page_title="Visual Intelligence - Smart Preprocessing",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# عنوان الصفحة
# -----------------------------
st.markdown("""
# :right_anger_bubble: Smart Image Preprocessing
""")

# -----------------------------
# لوحة جانبية للخيارات
# -----------------------------
st.sidebar.header(" إعدادات المعالجة")
apply_processing = st.sidebar.checkbox("تطبيق تحسين الصورة الذكي", value=True)
show_original = st.sidebar.checkbox("عرض الصورة الأصلية جنب المحسّنة", value=True)
st.sidebar.markdown("---")

# -----------------------------
# رفع الصورة
# -----------------------------
uploaded_file = st.file_uploader(" ارفع صورة (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    
    # تحليل الصورة
    reader = ImageReader(file_bytes)
    
    if not reader.valid:
        st.error("لا يمكن قراءة الصورة. تحقق من الملف.")
    else:
        bgr_img = reader.img

        # إذا تحسين الصورة غير مفعل → عرض الصورة الأصلية فقط
        if not apply_processing:
            st.subheader(" الصورة الأصلية")
            st.image(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            # المواصفات تظهر تدريجيًا على الصورة الأصلية
            h, w = reader.size()
            spec_text = f"أبعاد الصورة: {h} x {w}  \n" \
                        f"النسبة البعدية: {reader.aspect_ratio()}  \n" \
                        f"دقة الصورة: {reader.resolution()}"
            st.subheader("مواصفات الصورة")
            placeholder = st.empty()
            display_text = ""
            for char in spec_text:
                display_text += char
                placeholder.markdown(display_text)
                time.sleep(0.05)
        
        else:
            # تحسين الصورة
            enhancer = ImageEnhancerSmartQuality()
            enhanced_img = enhancer.enhance(bgr_img)

            # عرض الصور جنب بعض
            if show_original:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("الصورة الأصلي:round_pushpin:	")
                    st.image(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2:
                    st.subheader("الصورة المحسنة:round_pushpin:	")
                    st.image(enhanced_img, use_container_width=True)

            # خيار القرار للعمل على الصورة المحسنة
            st.subheader("اختيار العمل على الصورة")
            decision = st.radio(
                "هل تريد الاستمرار بالعمل على الصورة المحسّنة؟",
                ("اختر", "نعم", "لا")
            )

            if decision in ("نعم", "لا"):
                spec_img = enhanced_img if decision == "نعم" else bgr_img

                # المواصفات تظهر تدريجيًا بعد اختيار المستخدم
                h, w = reader.size()
                spec_text = f"أبعاد الصورة: {h} x {w}  \n" \
                            f"النسبة البعدية: {reader.aspect_ratio()}  \n" \
                            f"دقة الصورة: {reader.resolution()}"
                st.subheader(" مواصفات الصورة")
                placeholder = st.empty()
                display_text = ""
                for char in spec_text:
                    display_text += char
                    placeholder.markdown(display_text)
                    time.sleep(0.05)

                # زر تنزيل الصورة المختارة
                result_img = Image.fromarray(cv2.cvtColor(spec_img, cv2.COLOR_BGR2RGB))
                st.download_button(
                    " تنزيل الصورة:floppy_disk:	",
                    data=result_img.tobytes(),
                    file_name="selected_image.png",
                    mime="image/png"
                )
