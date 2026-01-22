import streamlit as st
from PIL import Image
import pytesseract
from docx import Document
from io import BytesIO
import os

# Configure page
st.set_page_config(
    page_title="OCR Text Extractor",
    page_icon="📝",
    layout="wide"
)

# Title and description
st.title("📝 OCR Text Extractor")
st.markdown("Upload images to extract text and download as a Word document")

# Sidebar for settings
st.sidebar.header("Settings")
language = st.sidebar.selectbox(
    "OCR Language",
    ["eng", "eng+ara", "eng+hin"],
    help="Select the language for text recognition"
)

# Initialize session state for extracted texts
if 'extracted_texts' not in st.session_state:
    st.session_state.extracted_texts = []

# File uploader
uploaded_files = st.file_uploader(
    "Upload Images",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=True,
    help="Upload one or more images containing text"
)

def extract_text_from_image(image):
    """Extract text from image using Tesseract OCR"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Perform OCR
        text = pytesseract.image_to_string(image, lang=language)
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def create_word_document(texts):
    """Create a Word document with extracted texts"""
    doc = Document()
    doc.add_heading('Extracted Text from Images', 0)
    
    for i, text in enumerate(texts, 1):
        doc.add_heading(f'Image {i}', level=1)
        doc.add_paragraph(text)
        doc.add_page_break()
    
    # Save to BytesIO
    doc_io = BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)
    return doc_io

# Main processing area
if uploaded_files:
    st.subheader(f"Processing {len(uploaded_files)} image(s)")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    extracted_texts = []
    
    for idx, uploaded_file in enumerate(uploaded_files):
        with col1:
            st.markdown(f"#### Image {idx + 1}: {uploaded_file.name}")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown(f"#### Extracted Text {idx + 1}")
            with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                extracted_text = extract_text_from_image(image)
                extracted_texts.append(extracted_text)
                
                # Display in text area
                st.text_area(
                    f"Text from {uploaded_file.name}",
                    extracted_text,
                    height=300,
                    key=f"text_{idx}"
                )
        
        st.divider()
    
    # Store in session state
    st.session_state.extracted_texts = extracted_texts
    
    # Download options
    st.subheader("📥 Download Options")
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        # Download as Word document
        doc_file = create_word_document(extracted_texts)
        st.download_button(
            label="📄 Download as Word Document",
            data=doc_file,
            file_name="extracted_text.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
    
    with col_btn2:
        # Download as plain text
        combined_text = "\n\n" + "="*50 + "\n\n".join(
            [f"IMAGE {i+1}\n{text}" for i, text in enumerate(extracted_texts)]
        )
        st.download_button(
            label="📝 Download as Text File",
            data=combined_text,
            file_name="extracted_text.txt",
            mime="text/plain",
            use_container_width=True
        )

else:
    st.info("👆 Upload one or more images to get started")
    
    # Show example
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Upload Images**: Click the upload button and select one or more images
        2. **Wait for Processing**: The app will automatically extract text from each image
        3. **Review**: Check the extracted text in the preview area
        4. **Download**: Choose to download as Word document or plain text file
        
        **Tips for better results:**
        - Use clear, high-resolution images
        - Ensure good lighting and contrast
        - Avoid blurry or skewed images
        - For handwritten text, ensure clear and legible writing
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>Built with Streamlit | Powered by Tesseract OCR</div>",
    unsafe_allow_html=True
)
