document.addEventListener('DOMContentLoaded', function() {
    // 날짜 기본값 설정
    const today = new Date();
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(today.getFullYear() - 1);
    
    document.getElementById('startDate').value = oneYearAgo.toISOString().split('T')[0];
    document.getElementById('endDate').value = today.toISOString().split('T')[0];
    
    // 모델 타입에 따라 선형 모델 옵션 표시/숨기기
    document.getElementById('modelType').addEventListener('change', function() {
        const linearModelOptions = document.getElementById('linearModelOptions');
        if (this.value === 'linear') {
            linearModelOptions.style.display = 'block';
        } else {
            linearModelOptions.style.display = 'none';
        }
    });
    
    // 페이지 로드 시 초기 상태 설정
    const modelSelect = document.getElementById('modelType');
    const linearModelOptions = document.getElementById('linearModelOptions');
    
    if (modelSelect.value === 'linear') {
        linearModelOptions.style.display = 'block';
    } else {
        linearModelOptions.style.display = 'none';
    }
    
    // 데이터 가져오기
    document.getElementById('fetchData').addEventListener('click', function() {
        const symbol = document.getElementById('symbol').value;
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;
        
        fetch('/api/fetch_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol,
                start_date: startDate,
                end_date: endDate
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 대시보드 표시
                const dashboardPath = data.data.dashboard_path;
                document.getElementById('dashboardFrame').src = dashboardPath;
                
                alert('Data fetched successfully!');
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while fetching data.');
        });
    });
    
    // 모델 학습
    document.getElementById('trainModels').addEventListener('click', function() {
        const symbol = document.getElementById('symbol').value;
        const modelType = document.getElementById('modelType').value;
        const linearType = document.getElementById('linearType').value;
        
        fetch('/api/train_models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol,
                model_type: modelType,
                linear_type: linearType
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 평가 결과 표시
                const evaluationPath = data.evaluation_path;
                document.getElementById('dashboardFrame').src = evaluationPath;
                
                alert('Models trained successfully!');
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while training models.');
        });
    });
    
    // 예측
    document.getElementById('predict').addEventListener('click', function() {
        const symbol = document.getElementById('symbol').value;
        const modelType = document.getElementById('modelType').value;
        const linearType = document.getElementById('linearType').value;
        
        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol,
                model_type: modelType,
                linear_type: linearType
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 예측 결과 표시
                const visualizationPath = data.visualization_path;
                document.getElementById('dashboardFrame').src = visualizationPath;
                
                alert('Prediction completed successfully!');
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while making predictions.');
        });
    });
}); 