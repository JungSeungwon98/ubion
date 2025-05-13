
import os 
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_core.documents import Document
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import streamlit as st
import fitz
import tempfile
import uuid
import plotly.graph_objects as go
import re

@st.cache_resource
def load_ko_embedding():
    # 이 함수 안에서만 torch와 HuggingFace를 임포트
    
    
    if not torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    model_name = "jhgan/ko-sbert-nli"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
## 1: 임시폴더에 파일 저장
def save_uploadedfile(uploadedfile: UploadedFile) -> str : 
    temp_dir = "PDF_임시폴더"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.read()) 
    return file_path

# 저장된 pdf를 document로 변환 저장 
def pdf_to_documents(pdf_path:str) -> List[Document]:
    # documents_list = []
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load_and_split()
    # print(documents)
    
    return documents

#더 작은 단위의 chunk로 변환
def chunck_documents(documents:List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    chuk_list = text_splitter.split_documents(documents)
    # print(len(documents))
    # print(len(chuk_list))
    return chuk_list

# FAISS 벡터DB에 저장
def save_to_vector_store(chuk_list:List[Document]) -> None:
    ko_embeddings = load_ko_embedding()
    vector_store = FAISS.from_documents(chuk_list , embedding=ko_embeddings)
    vector_store.save_local("faiss_index")

# @st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)  # 문서 열기
    image_paths = []
    
    # 이미지 저장용 폴더 생성 (시스템 임시 디렉토리 사용)
    output_folder = os.path.join(tempfile.gettempdir(), "PDF_이미지")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for page_num in range(len(doc)):  #  각 페이지를 순회
        page = doc.load_page(page_num)  # 페이지 로드

        zoom = dpi / 72  # 72이 디폴트 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat) # type: ignore

        image_filename = f"page_{page_num + 1}.png"
        image_path = os.path.join(output_folder, image_filename)
        
        # 기존 파일 삭제 시도
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                # 권한 오류 발생 시 대체 파일명 사용
                image_filename = f"{uuid.uuid4().hex}_page_{page_num + 1}.png"
                image_path = os.path.join(output_folder, image_filename)
        
        try:
            pix.save(image_path)  # PNG 형태로 저장
        except PermissionError:
            # 권한 오류 발생 시 대체 파일명 사용
            image_filename = f"{uuid.uuid4().hex}_page_{page_num + 1}.png"
            image_path = os.path.join(output_folder, image_filename)
            pix.save(image_path)
            
        image_paths.append(image_path)  # 경로를 저장
        
    return image_paths

def display_pdf_page(image_path: str, page_number: int) -> None:
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(image_path):
            st.error(f"이미지 파일이 존재하지 않습니다.")
            return
            
        with open(image_path, "rb") as f:
            image_bytes = f.read()  # 파일에서 이미지 인식
        
        st.image(image_bytes, use_container_width=True)
    except Exception as e:
        st.error(f"이미지 표시 중 오류가 발생했습니다.")

def create_image_figure(image_path: str) -> go.Figure:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=image_bytes,
            x=0,
            y=1,
            xref="paper",
            yref="paper",
            sizex=1,
            sizey=1,
            sizing="stretch",
            layer="below"
        )
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]    
    
# documents = pdf_to_documents("PDF_임시폴더\SPRi AI Brief_4월호_산업동향__250407_F.pdf")
# # print(documents)
# chuk_list = chunck_documents(documents)
# save_to_vector_store(chuk_list)

# convert_pdf_to_images("PDF_임시폴더/SPRi AI Brief_4월호_산업동향__250407_F.pdf")
