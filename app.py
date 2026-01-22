import streamlit as st
import cv2
import pytesseract
from PIL import Image
from docx import Document

# FUNCTIONS
def extract_text(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_string(img)

def save_diagram(image_path, diagram_path):
    img = cv2.imread(image_path)
    cv2.imwrite(diagram_path, img)

def create_document(image_paths, output_docx="output.docx"):
    doc = Document()
    for idx, img_path in enumerate(image_paths):
        text = extract_text(img_path)
        doc.add_heading(f"Image {idx+1}", level=1)
        doc.add_paragraph(text)
        diagram_file = f"diagram_{idx+1}.png"
        save_diagram(img_path, diagram_file)
        doc.add_picture(diagram_file)
    doc.save(output_docx)
    return output_docx

# STREAMLIT UI
st.title("🖼️ Image to Editable Document App")

uploaded_files = st.file_uploader(
    "Upload exactly 3 images", type=["png","jpg","jpeg"], accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) == 3:
    st.write("Processing images...")
    
    image_paths = []
    for i, uploaded_file in enumerate(uploaded_files):
        img_path = f"image_{i+1}.png"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_paths.append(img_path)
    
    output_docx = create_document(image_paths)
    st.success("✅ Document created successfully!")
    
    with open(output_docx, "rb") as f:
        st.download_button("📥 Download Document", f, file_name=output_docx)
else:
    st.info("Please upload exactly 3 images.")
