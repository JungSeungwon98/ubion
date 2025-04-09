
document.addEventListener('DOMContentLoaded', function() {
    // 날짜 기본값 설정
    const today = new Date();
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(today.getFullYear() - 1);
    
    document.getElementById('startDate').value = oneYearAgo.toISOString().split('T')[0];
    document.getElementById('endDate').value = today.toISOString().split('T')[0];
    
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
        
        fetch('/api/train_models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol
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
        
        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol,
                model_type: modelType
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 예측 결과 표시
                const visualizationPath = data.prediction_path;
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
        