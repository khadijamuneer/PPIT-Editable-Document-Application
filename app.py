import streamlit as st
import requests
from docx import Document
import io

st.title("📄 Image to Editable Document App")

uploaded_files = st.file_uploader(
    "Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

API_KEY = "helloworld"  # Free OCR.Space test key

def ocr_space_file(image_file):
    """Send image to OCR.Space API and return extracted text"""
    try:
        files = {"filename": image_file.getvalue()}
        payload = {"apikey": API_KEY, "OCREngine": 2, "isTable": True}
        response = requests.post("https://api.ocr.space/parse/image", files=files, data=payload)
        result = response.json()
        if "ParsedResults" in result and result["ParsedResults"]:
            return result["ParsedResults"][0]["ParsedText"]
        else:
            return "[No text detected or OCR failed]"
    except Exception as e:
        return f"[Error: {str(e)}]"

if uploaded_files:
    doc = Document()
    for idx, uploaded_file in enumerate(uploaded_files):
        doc.add_heading(f"Image {idx+1}", level=1)
        text = ocr_space_file(uploaded_file)
        doc.add_paragraph(text)

    doc.save("output.docx")
    st.success("✅ Document created successfully!")
    with open("output.docx", "rb") as f:
        st.download_button("Download Document", f, file_name="output.docx")
else:
    st.info("Please upload at least one image.")
