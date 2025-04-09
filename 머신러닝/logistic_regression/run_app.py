import os
import sys
import subprocess
import platform

def check_streamlit_installed():
    """Streamlit이 설치되어 있는지 확인합니다."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_requirements():
    """필요한 패키지를 설치합니다."""
    print("필요한 패키지를 설치합니다...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError:
        return False

def run_streamlit():
    """Streamlit 앱을 실행합니다."""
    # Streamlit이 설치되어 있는지 확인
    if not check_streamlit_installed():
        print("Streamlit이 설치되어 있지 않습니다. 설치를 시도합니다...")
        if not install_requirements():
            print("패키지 설치에 실패했습니다. 수동으로 설치해주세요:")
            print("pip install -r requirements.txt")
            return
    
    # 운영체제에 따라 명령어 구성
    if platform.system() == "Windows":
        cmd = f"{sys.executable} -m streamlit run app.py"
    else:
        cmd = f"streamlit run app.py"
    
    print("Streamlit 앱을 시작합니다...")
    print(f"실행 명령어: {cmd}")
    print("브라우저에서 http://localhost:8501 로 접속하세요.")
    
    # 명령어 실행
    os.system(cmd)

if __name__ == "__main__":
    run_streamlit() 