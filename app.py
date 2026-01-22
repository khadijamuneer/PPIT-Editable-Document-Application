import streamlit as st
import requests
from docx import Document
import io

st.title("📄 Image to Editable Document App")

uploaded_files = st.file_uploader(
    "Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

API_KEY = "helloworld"  # Free API key from OCR.Space

def ocr_space_file(image_bytes):
    """Send image to OCR.Space and return extracted text"""
    response = requests.post(
        "https://api.ocr.space/parse/image",
        files={"filename": image_bytes},
        data={"apikey": API_KEY, "OCREngine": 2, "isTable": True},
    )
    result = response.json()
    if result.get("ParsedResults"):
        return result["ParsedResults"][0]["ParsedText"]
    else:
        return ""

if uploaded_files:
    doc = Document()
    for idx, uploaded_file in enumerate(uploaded_files):
        doc.add_heading(f"Image {idx+1}", level=1)

        image_bytes = uploaded_file.read()
        text = ocr_space_file(io.BytesIO(image_bytes))
        if text.strip() == "":
            doc.add_paragraph("[No text detected]")
        else:
            doc.add_paragraph(text)

    doc.save("output.docx")
    st.success("✅ Document created successfully!")
    with open("output.docx", "rb") as f:
        st.download_button("Download Document", f, file_name="output.docx")
else:
    st.info("Please upload at least one image.")
