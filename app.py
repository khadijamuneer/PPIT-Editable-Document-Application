import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
from docx import Document
from PIL import Image
import io

st.set_page_config(page_title="Image to Editable Document")

st.title("📄 Image to Editable Document App")

uploaded_files = st.file_uploader(
    "Upload 3 images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) == 3:
    # Google Vision credentials from Streamlit secrets
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcloud_service_account"]
    )
    client = vision.ImageAnnotatorClient(credentials=credentials)

    doc = Document()

    for idx, uploaded_file in enumerate(uploaded_files):
        doc.add_heading(f"Image {idx+1}", level=1)

        # Read image bytes
        image_bytes = uploaded_file.read()
        image = vision.Image(content=image_bytes)

        # OCR using Google Vision
        response = client.text_detection(image=image)
        text = response.text_annotations[0].description if response.text_annotations else ""
        doc.add_paragraph(text)

        # Add image to Word doc
        img = Image.open(io.BytesIO(image_bytes))
        img_path = f"image_{idx+1}.png"
        img.save(img_path)
        doc.add_picture(img_path)

    # Save Word doc
    doc.save("output.docx")
    st.success("✅ Document created successfully!")
    with open("output.docx", "rb") as f:
        st.download_button("Download Document", f, file_name="output.docx")
else:
    st.info("Please upload exactly 3 images.")
