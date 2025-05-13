import streamlit as st
from utils import save_uploadedfile

def main():
    st.header("FAQ 챗봇")
    pdf_doc = st.file_uploader("PDF 업로더", type="pdf")
    button = st.button("pdf 업로드하기", type="primary")
    st.write(button)
    if button and pdf_doc:
        pdf_path = save_uploadedfile(pdf_doc)
        st.write(pdf_path)
    else:
        st.write("No pdf upload")

    
if __name__ == "__main__":
    main()
    