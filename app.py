import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import easyocr
from docx import Document
from io import BytesIO
import numpy as np
import cv2

st.set_page_config(page_title="Advanced OCR Extractor", page_icon="📝", layout="wide")

st.title("📝 Advanced OCR Text Extractor - Maximum Accuracy")
st.markdown("Extracts text from images with advanced preprocessing for best results")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    ocr_lang = st.multiselect(
        "Language",
        ["en", "hi", "ar"],
        default=["en"]
    )
    
    st.subheader("🔧 Enhancement Level")
    enhancement = st.select_slider(
        "Choose enhancement level",
        options=["Minimal", "Light", "Medium", "Strong", "Maximum"],
        value="Medium"
    )
    
    use_ai_correction = st.checkbox(
        "Use AI Text Correction (Experimental)",
        value=False,
        help="Uses AI to fix common OCR errors - slower but more accurate"
    )

@st.cache_resource
def load_reader(langs):
    return easyocr.Reader(langs, gpu=False)

def advanced_preprocessing(img, level):
    """Advanced multi-stage preprocessing"""
    # Convert to grayscale
    if img.mode != 'L':
        img = ImageOps.grayscale(img)
    
    img_array = np.array(img)
    
    # Stage 1: Noise Reduction
    if level in ["Medium", "Strong", "Maximum"]:
        img_array = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
    
    # Stage 2: Enhance contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if level in ["Strong", "Maximum"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_array = clahe.apply(img_array)
    
    # Stage 3: Morphological operations to clean up
    if level == "Maximum":
        kernel = np.ones((2,2), np.uint8)
        img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
    
    # Stage 4: Adaptive thresholding
    if level in ["Light", "Medium"]:
        img_array = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    elif level in ["Strong", "Maximum"]:
        img_array = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 3
        )
    
    # Stage 5: Deskew (correct slight rotation)
    if level in ["Strong", "Maximum"]:
        coords = np.column_stack(np.where(img_array > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            if abs(angle) > 0.5:  # Only deskew if angle is significant
                (h, w) = img_array.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img_array = cv2.warpAffine(
                    img_array, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
    
    # Stage 6: Final sharpening
    if level in ["Medium", "Strong", "Maximum"]:
        pil_img = Image.fromarray(img_array)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
        img_array = np.array(pil_img)
    
    # Convert back to PIL
    return Image.fromarray(img_array)

def smart_text_extraction(img, reader):
    """Extract text with smart layout detection"""
    img_array = np.array(img)
    
    # Get results with bounding boxes
    results = reader.readtext(img_array, paragraph=False, detail=1)
    
    if not results:
        return "⚠️ No text detected"
    
    # Sort by position (top to bottom, left to right)
    results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
    
    # Group into lines
    lines = []
    current_line = []
    prev_y = None
    
    for detection in results:
        bbox, text, confidence = detection
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        
        # New line if y position differs significantly
        if prev_y is None or abs(y_center - prev_y) > 20:
            if current_line:
                lines.append(current_line)
            current_line = [detection]
            prev_y = y_center
        else:
            current_line.append(detection)
    
    if current_line:
        lines.append(current_line)
    
    # Build text with proper formatting
    output = []
    for line in lines:
        # Sort words in line by x position
        line = sorted(line, key=lambda x: x[0][0][0])
        
        # Extract text and add spacing
        words = []
        prev_x = None
        for bbox, text, conf in line:
            x_start = bbox[0][0]
            
            # Add extra space if gap is large
            if prev_x and (x_start - prev_x) > 50:
                words.append("  ")
            
            words.append(text)
            prev_x = bbox[1][0]  # Right edge
        
        output.append(' '.join(words))
    
    return '\n'.join(output)

def ai_text_correction(text):
    """Use Claude API to correct OCR errors"""
    try:
        import anthropic
        
        # Check if API key is available
        if 'ANTHROPIC_API_KEY' not in st.secrets:
            return text, "API key not configured"
        
        client = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])
        
        prompt = f"""You are an OCR text correction assistant. The following text was extracted from a handwritten chemistry note using OCR and may contain errors.

Please correct:
1. Spelling errors
2. Common OCR mistakes (l→I, 0→O, etc.)
3. Chemical formulas (H2O → H₂O, etc.)
4. Scientific terms

Preserve the original formatting and line breaks. Only fix errors, don't rewrite.

Text to correct:
{text}

Return ONLY the corrected text, nothing else."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        corrected = message.content[0].text
        return corrected, "Success"
        
    except Exception as e:
        return text, f"AI correction failed: {str(e)}"

def create_document(texts, filenames):
    """Create Word document"""
    doc = Document()
    doc.add_heading('Extracted Text', 0)
    
    for i, (text, name) in enumerate(zip(texts, filenames), 1):
        doc.add_heading(f'Image {i}: {name}', 1)
        doc.add_paragraph(text)
        if i < len(texts):
            doc.add_page_break()
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# File uploader
uploaded_files = st.file_uploader(
    "Upload Images",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
    
    # Load OCR
    with st.spinner("Loading OCR engine..."):
        reader = load_reader(ocr_lang)
    
    all_texts = []
    all_names = []
    
    for idx, file in enumerate(uploaded_files):
        st.markdown(f"## 📄 Image {idx + 1}: {file.name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            original = Image.open(file)
            st.image(original, use_container_width=True)
            
            st.subheader("Preprocessed")
            processed = advanced_preprocessing(original, enhancement)
            st.image(processed, use_container_width=True)
        
        with col2:
            st.subheader("Extracted Text")
            
            with st.spinner("Extracting text..."):
                # Extract text
                extracted = smart_text_extraction(processed, reader)
                
                # AI correction if enabled
                if use_ai_correction and not extracted.startswith("⚠️"):
                    with st.spinner("Applying AI correction..."):
                        extracted, status = ai_text_correction(extracted)
                        if status != "Success":
                            st.warning(f"AI correction: {status}")
                
                # Editable text area
                final_text = st.text_area(
                    "Review and edit:",
                    extracted,
                    height=500,
                    key=f"text_{idx}"
                )
                
                all_texts.append(final_text)
                all_names.append(file.name)
                
                # Stats
                words = len(final_text.split())
                chars = len(final_text)
                lines = len(final_text.splitlines())
                st.caption(f"📊 {words} words • {chars} characters • {lines} lines")
        
        st.divider()
    
    # Download section
    st.markdown("## 📥 Download")
    
    col1, col2 = st.columns(2)
    
    with col1:
        doc = create_document(all_texts, all_names)
        st.download_button(
            "📄 Download Word Document",
            doc,
            "extracted_text.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
    
    with col2:
        combined = "\n\n".join([
            f"{'='*60}\nIMAGE {i+1}: {name}\n{'='*60}\n\n{text}"
            for i, (name, text) in enumerate(zip(all_names, all_texts))
        ])
        st.download_button(
            "📝 Download Text File",
            combined,
            "extracted_text.txt",
            use_container_width=True
        )

else:
    st.info("👆 Upload images to start")
    
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        ### 📸 Image Quality Tips:
        - Use good lighting
        - Keep camera steady (no blur)
        - Ensure text fills most of the frame
        - Avoid shadows and glare
        - Higher resolution is better
        
        ### ⚙️ Settings Guide:
        - **Minimal**: For very clear, dark text
        - **Light**: For good quality handwriting
        - **Medium**: For average handwriting (recommended)
        - **Strong**: For faded or light writing
        - **Maximum**: For difficult/poor quality images
        
        ### 🤖 AI Correction:
        - Enable for better accuracy
        - Requires Anthropic API key in secrets
        - Slower but corrects common OCR errors
        - Best for technical content
        """)

st.markdown("---")
st.caption("Built with EasyOCR | Advanced preprocessing for maximum accuracy")
