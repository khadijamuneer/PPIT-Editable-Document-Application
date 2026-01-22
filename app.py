import streamlit as st
from PIL import Image
from docx import Document
import easyocr

# Initialize OCR reader (English)
reader = easyocr.Reader(['en'])

# FUNCTIONS
def extract_text(image_path):
    img = Image.open(image_path)
    results = reader.readtext(img)
    text = "\n".join([res[1] for res in results])
    return text

def save_diagram(image_path, diagram_path):
    img = Image.open(image_path)
    img.save(diagram_path)

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
