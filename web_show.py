# file: app_with_objects.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os

from processors import ImageEnhancerSmartQuality, ImageReader, analyze_image_file

st.set_page_config(
    page_title="Visual Intelligence - Smart Preprocessing",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("# :right_anger_bubble: Smart Image Preprocessing")

# Sidebar
st.sidebar.header("إعدادات المعالجة")
apply_processing = st.sidebar.checkbox("تطبيق تحسين الصورة الذكي", value=True)
show_original = st.sidebar.checkbox("عرض الصورة الأصلية جنب المحسّنة", value=True)
st.sidebar.markdown("---")

# Upload
uploaded_file = st.file_uploader("ارفع صورة (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    reader = ImageReader(file_bytes)
    
    if not reader.valid:
        st.error("لا يمكن قراءة الصورة. تحقق من الملف.")
    else:
        bgr_img = reader.img
        enhanced_img = ImageEnhancerSmartQuality().enhance(bgr_img) if apply_processing else bgr_img

        # عرض الصور
        if show_original and apply_processing:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("الصورة الأصلية :round_pushpin:")
                st.image(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader("الصورة المحسنة :round_pushpin:")
                st.image(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.image(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        # اختيار العمل على الصورة
        st.subheader("اختيار العمل على الصورة")
        decision = st.radio("هل تريد الاستمرار بالعمل على الصورة المحسّنة؟", ("اختر", "نعم", "لا"))
        if decision in ("نعم", "لا"):
            spec_img = enhanced_img if decision=="نعم" else bgr_img

            # مواصفات الصورة
            h, w = reader.size()
            st.subheader("مواصفات الصورة")
            st.markdown(f"""
            - أبعاد الصورة: {h} x {w}
            - النسبة البعدية: {reader.aspect_ratio()}
            - دقة الصورة: {reader.resolution()}
            """)

            # تحليل الكائنات
            try:
                result = analyze_image_file(file_bytes, output_dir="outputs", enhance_first=False)
                objects = result.get("objects", [])
                arabic_caption = result.get("arabic_caption", "—")

                st.subheader("الكائنات المكتشفة")
                if not objects:
                    st.info("لا توجد كائنات مكتشفة في الصورة.")
                else:
                    for obj in objects:
                        label = obj.get("label", "كائن مجهول")
                        colors = obj.get("dominant_colors", [])
                        bbox = obj.get("bbox", {})
                        x1, y1, x2, y2 = bbox.get("x1",0), bbox.get("y1",0), bbox.get("x2",0), bbox.get("y2",0)
                        roi = spec_img[y1:y2, x1:x2].copy()
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        roi_pil = Image.fromarray(roi_rgb)

                        st.markdown(f"### {label}")
                        # عرض الصورة المصغرة للكائن
                        st.image(roi_pil, width=150)

                        # عرض الألوان الأساسية للكائن
                        if len(colors) > 0:
                            cols = st.columns(len(colors))
                            for i, c in enumerate(colors):
                                hex_color = c["hex"]
                                cols[i].markdown(
                                    f"""
                                    <div style='
                                        background-color:{hex_color};
                                        width:50px;
                                        height:50px;
                                        border-radius:8px;
                                        margin:auto;
                                        border:2px solid #fff;
                                    '></div>
                                    """,
                                    unsafe_allow_html=True
                                )

                # وصف الصورة العام
                st.subheader("وصف الصورة")
                st.markdown(arabic_caption)

            except Exception as e:
                st.error(f"فشل تحليل الصورة: {e}")

            # زر تنزيل الصورة
            result_img = Image.fromarray(cv2.cvtColor(spec_img, cv2.COLOR_BGR2RGB))
            st.download_button(
                "تنزيل الصورة :floppy_disk:",
                data=result_img.tobytes(),
                file_name="selected_image.png",
                mime="image/png"
            )
