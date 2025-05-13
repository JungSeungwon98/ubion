"""
PDF 문서 질의응답 시스템 (CLI 버전)
"""
import argparse
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
from datetime import datetime
import webbrowser
import plotly.graph_objects as go
import base64
import sys
from langchain_core.language_models.llms import LLM
from typing import Any, List, Mapping, Optional
import load_dotenv


load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘
    - 질문에 대한 응답을 해줘
    - 간결하게 5줄 이내로 해줘
    - 곧바로 응답결과를 말해줘

    컨텍스트 : {context}

    질문: {question}

    응답:
    
    """
prompt_template = PromptTemplate.from_template(template)


def load_korean_model(model_name: str):
    """한국어 모델을 로드하는 함수"""
    try:
        model_configs = {
            "KoBART": {
                "name": "gogamza/kobart-base-v2",
                "max_length": 128,
                "temperature": 0.7,
                "top_p": 0.95,
                "repetition_penalty": 1.15
            }
        }

        model_config = model_configs.get(model_name)
        if not model_config:
            raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
        
        print(f"{model_name} 모델 로딩 중...")
        print(f"모델 크기: {model_config['name']}")
        print(f"최대 토큰 길이: {model_config['max_length']}")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["name"],
            trust_remote_code=True
        )
        
        # 모델 로드 (CPU 사용)
        model = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 파이프라인 설정
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=model_config["max_length"],
            temperature=model_config["temperature"],
            top_p=model_config["top_p"],
            repetition_penalty=model_config["repetition_penalty"],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        print(f"{model_name} 모델 로딩 완료!")
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # 모델이 제대로 로드되었는지 확인
        if llm is None:
            raise ValueError(f"{model_name} 모델이 올바르게 로드되지 않았습니다.")
            
        return llm
    except Exception as e:
        print(f"{model_name} 모델 로드 중 오류 발생: {str(e)}")
        # 오류 발생 시 대체 모델 사용 (OpenAI 또는 다른 백업 모델)
        print("기본 모델로 대체합니다...")
        return ChatOpenAI(temperature=0.7, max_tokens=200)

def main():
    # utils 모듈에서 임베딩 로드
    from utils import load_ko_embedding
    
    # 모델 로드
    llm_model = load_korean_model("KoBART")
    openai_api_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=200)
    # 모델이 제대로 로드되었는지 확인
    if llm_model is None:
        print("모델 로드에 실패했습니다. 프로그램을 종료합니다.")
        sys.exit(1)
    
    try:
        # FAISS 인덱스 로드
        db = FAISS.load_local(
            "faiss_index",
            embeddings=load_ko_embedding(),
            allow_dangerous_deserialization=True
        )
        
        # RetrievalQA 체인 생성
        qa = RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type="stuff",
            retriever=db.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 3, 'fetch_k': 10}
            ),
            chain_type_kwargs={'prompt': prompt_template},
            return_source_documents=True
        )
        
        openai_qa = RetrievalQA.from_chain_type(
            llm=openai_api_model,
            chain_type="stuff",
            retriever=db.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 3, 'fetch_k': 10}
            ),
            chain_type_kwargs={'prompt': prompt_template},
            return_source_documents=True
        )
        # 쿼리 실행
        response = qa({
            "query": "AI 산업동향은?"
        })
        
        response_1 = openai_qa({
            "query": "AI 산업동향은?"
        })
        
        # print(response)
        print("=="*10)
        print(response_1)
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()