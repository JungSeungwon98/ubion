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

# 환경변수 로드
load_dotenv()

# HuggingFace 임베딩 사용 준비
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘
    - 질문에 대한 응답을 해줘
    - 간결하게 5줄 이내로 해줘
    - 곧바로 응답결과를 말해줘

    컨텍스트 : {context}

    질문: {question}

    응답:"""

# 템플릿을 PromptTemplate 객체로 변환
prompt_template = PromptTemplate.from_template(template)

def load_ko_embedding():
    """한국어 임베딩 모델을 로드하는 함수"""
    model_name = "jhgan/ko-sbert-nli"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def load_korean_model(model_name: str):
    """한국어 모델을 로드하는 함수"""
    try:
        # 모델별 설정
        model_configs = {
            # 대규모 모델
            "KoAlpaca-Polyglot": {
                "name": "beomi/KoAlpaca-Polyglot-12.8B",
                "max_length": 2048,
                "temperature": 0.7,
                "top_p": 0.95,
                "repetition_penalty": 1.15
            },
            # 가벼운 모델
            "KoBART": {
                "name": "gogamza/kobart-base-v2",
                "max_length": 128,
                "temperature": 0.7,
                "top_p": 0.95,
                "repetition_penalty": 1.15
            },
            "KoBART-Small": {
                "name": "gogamza/kobart-small-v2",
                "max_length": 128,
                "temperature": 0.7,
                "top_p": 0.95,
                "repetition_penalty": 1.15
            },
            "KoGPT-Small": {
                "name": "skt/kogpt2-base-v2",
                "max_length": 128,
                "temperature": 0.7,
                "top_p": 0.95,
                "repetition_penalty": 1.15
            }
        }
        
        config = model_configs.get(model_name)
        if not config:
            raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
        
        print(f"{model_name} 모델 로딩 중...")
        print(f"모델 크기: {config['name']}")
        print(f"최대 토큰 길이: {config['max_length']}")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            config["name"],
            trust_remote_code=True
        )
        
        # 모델 로드 (CPU 사용)
        model = AutoModelForCausalLM.from_pretrained(
            config["name"],
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 파이프라인 설정
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config["max_length"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            repetition_penalty=config["repetition_penalty"],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        print(f"{model_name} 모델 로딩 완료!")
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        print(f"{model_name} 모델 로드 중 오류 발생: {str(e)}")
        return None

def get_llm(model_option: str):
    """선택된 모델에 따라 LLM을 반환하는 함수"""
    try:
        if model_option == "Gemini 2.0 Flash":
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("Gemini 2.0 flash를 사용하려면 GOOGLE_API_KEY가 필요합니다.")
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                convert_system_message_to_human=True,
                google_api_key=os.environ.get("GOOGLE_API_KEY"),
                max_output_tokens=2048,
                top_p=0.8,
                top_k=40
            )
        elif model_option in ["KoAlpaca-Polyglot",  "KoBART", "KoBART-Small", "KoGPT-Small"]:
            return load_korean_model("KoBART")
        else:
            model_name = "gpt-4" if model_option == "GPT-4" else "gpt-3.5-turbo"
            return ChatOpenAI(
                model_name=model_name,
                temperature=0
            )
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {str(e)}")
        return None

def process_question(user_question: str, model_option: str):
    """사용자 질문을 처리하는 함수"""
    try:
        # 임베딩 모델 로드
        embeddings = load_ko_embedding()
        if not embeddings:
            raise ValueError("임베딩 모델 로드에 실패했습니다.")

        # 기존 FAISS 인덱스 로드
        if not os.path.exists("faiss_index"):
            raise ValueError("FAISS 인덱스가 없습니다. 먼저 PDF를 처리해주세요.")
        
        print("FAISS 인덱스 로드 중...")
        db = FAISS.load_local(
            "faiss_index", 
            embeddings,
            allow_dangerous_deserialization=True  # 보안 경고 무시 옵션 추가
        )
        print("FAISS 인덱스 로드 완료")

        # LLM 로드
        print(f"{model_option} 모델 로드 중...")
        llm = get_llm(model_option)
        if not llm:
            raise ValueError("모델 로드에 실패했습니다.")
        print("모델 로드 완료")

        # RetrievalQA 체인 생성
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 3, 'fetch_k': 10}
            ),

            
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )

        # 질문 처리
        print("질문 처리 중...")
        try:
            # invoke 대신 __call__ 사용
            result = qa({"query": user_question})
        except Exception as e:
            print(f"질문 처리 중 오류 발생: {str(e)}")
            # 대체 방법 시도
            result = qa.run(user_question)
            if not result:
                raise ValueError("질문 처리에 실패했습니다.")
            # 결과 형식 맞추기
            result = {
                "result": result,
                "source_documents": []
            }
        
        print("질문 처리 완료")
        return result
    except Exception as e:
        print(f"질문 처리 중 오류 발생: {str(e)}")
        return None

def create_image_figure(image_path: str, title: str) -> go.Figure:
    """이미지를 Plotly figure로 변환하는 함수"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        fig = go.Figure()
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{encoded_string}",
                xref="paper",
                yref="paper",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                sizing="stretch",
                layer="below"
            )
        )
        
        fig.update_layout(
            title=title,
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            margin=dict(l=0, r=0, t=30, b=0),
            height=800
        )
        
        return fig
    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {str(e)}")
        return None

def save_results_to_html(result, output_file: str):
    """결과를 HTML 파일로 저장하는 함수"""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            html_template = f"""
<html>
<head>
    <title>RAG 결과</title>
    <style>
        body {{ font-family: 'Malgun Gothic', '맑은 고딕', sans-serif; margin: 20px; }}
        .answer {{ background-color: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .document {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>RAG 시스템 결과</h1>
    <div class="answer">
        <h2>질문에 대한 답변</h2>
        <p>{str(result['result'])}</p>
    </div>
"""
            f.write(html_template)

            for i, doc in enumerate(result['source_documents'], 1):
                doc_template = f"""
    <div class="document">
        <h3>문서 {i}</h3>
        <p><strong>내용:</strong> {doc.page_content}</p>
        <p><strong>출처:</strong> {doc.metadata.get('source', 'N/A')}</p>
        <p><strong>페이지:</strong> {doc.metadata.get('page', 'N/A')}</p>
"""
                f.write(doc_template)
                
                # PDF 이미지 표시
                source = doc.metadata.get('source', '')
                page = doc.metadata.get('page', '')
                if source and page:
                    pdf_name = os.path.basename(source)
                    image_name = f"page_{page}.png"
                    image_path = os.path.join("PDF_이미지", image_name)
                    
                    if os.path.exists(image_path):
                        fig = create_image_figure(image_path, f"문서 {i}의 참조 페이지")
                        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
                
                f.write("    </div>\n")
            
            f.write("""
</body>
</html>
""")
        
        print(f"\n결과가 {output_file} 파일에 저장되었습니다.")
        webbrowser.open(f"file://{os.path.abspath(output_file)}")
    except Exception as e:
        print(f"결과 저장 중 오류 발생: {str(e)}")
        raise  # 오류 상세 정보를 확인하기 위해 예외를 다시 발생시킴

def main():
    parser = argparse.ArgumentParser(description="PDF 문서 질의응답 시스템")
    parser.add_argument("--question", "-q", required=True, help="질문을 입력하세요")
    parser.add_argument("--model", "-m", default="GPT-3.5 Turbo",
                      choices=["GPT-3.5 Turbo", "GPT-4", "Gemini 2.0 Flash", "KoBART"],
                      help="사용할 AI 모델을 선택하세요")
    parser.add_argument("--output", "-o", help="결과를 저장할 HTML 파일 경로")
    
    args = parser.parse_args()
    
    # 결과 파일 경로 설정
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results_{timestamp}.html"
    
    # 질문 처리
    result = process_question(args.question, args.model)
    if result:
        # 결과 출력
        print("\n=== 질문에 대한 답변 ===")
        print(result['result'])
        
        if 'source_documents' in result and result['source_documents']:
            print("\n=== 참조 문서 ===")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"\n[문서 {i}]")
                print(f"내용: {doc.page_content}")
                print(f"출처: {doc.metadata.get('source', 'N/A')}")
                print(f"페이지: {doc.metadata.get('page', 'N/A')}")
        
        # 결과를 HTML로 저장
        save_results_to_html(result, args.output)
    else:
        print("질문 처리에 실패했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()