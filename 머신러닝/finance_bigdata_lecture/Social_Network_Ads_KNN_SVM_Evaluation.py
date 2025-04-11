# -*- coding: utf-8 -*-
"""
예제: 첨부된 Social_Network_Ads_outlier_removed.csv 데이터를 활용하여
KNN 및 SVM 알고리즘을 이용한 분류 모델 평가 코드.
이 코드는 정확도, 혼동 행렬, 분류 보고서, 정밀도, 재현율, F1-score, 
교차 검증, ROC 커브, 러닝 커브 등을 포함합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize, LabelEncoder

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, feature_names):
    # 모델 학습
    model.fit(X_train, y_train)
    # 예측 수행
    y_pred = model.predict(X_test)
    
    # 1. Accuracy (정확도)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.3f}")
    
    # 2. Confusion Matrix (혼동 행렬)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{model_name} Confusion Matrix:\n{cm}")
    
    # 3. Classification Report (분류 보고서)
    report = classification_report(y_test, y_pred)
    print(f"\n{model_name} Classification Report:\n{report}")
    
    # 4. Precision, Recall, F1-score (macro 평균)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"{model_name} Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")
    
    # 5. Cross Validation Scores (5-겹 교차 검증)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{model_name} 5-Fold Cross Validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # 6. ROC AUC 및 ROC Curve (다중 클래스의 경우 one-vs-rest 방식)
    classes = np.unique(y_train)
    y_test_bin = label_binarize(y_test, classes=classes)
    # 이진 분류의 경우 (n_samples,1) 배열을 (n_samples,2)로 확장
    if y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))
    
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
        try:
            roc_auc = roc_auc_score(y_test_bin, y_score, multi_class='ovr', average='macro')
            print(f"{model_name} ROC AUC (macro, one-vs-rest): {roc_auc:.3f}")
        except ValueError as e:
            print(f"ROC AUC 계산 중 오류: {e}")
        
        # 각 클래스별 ROC Curve 그리기
        plt.figure()
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc_i = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"Class {classes[i]} (AUC = {roc_auc_i:.3f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curves')
        plt.legend(loc="lower right")
        plt.show()
    
    # 7. Learning Curve 그리기
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Cross-validation score")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.title(f'{model_name} Learning Curve')
    plt.legend(loc="best")
    plt.show()
    
    # 8. Feature Importances (특성 중요도) - SVM과 KNN은 지원하지 않으므로 건너뜁니다.
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure()
        plt.title(f"{model_name} Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.xlim([-1, len(importances)])
        plt.tight_layout()
        plt.show()

def main():
    # Social_Network_Ads_outlier_removed.csv 데이터 로드
    df = pd.read_csv('data/Social_Network_Ads_outlier_removed.csv', encoding='utf-8', index_col=0)
    
    # 범주형 변수 처리: 'Gender' 컬럼이 존재하면 숫자로 변환
    label_encoder = LabelEncoder()
    if 'Gender' in df.columns:
        df['Gender'] = label_encoder.fit_transform(df['Gender'])
    
    print("데이터 미리보기:")
    print(df.head())
    
    # 특성과 타겟 분리 (타겟 변수는 'Purchased'로 가정)
    X = df.drop('Purchased', axis=1)
    y = df['Purchased']
    feature_names = X.columns
    
    # 학습/테스트 데이터 분할 (테스트 데이터 30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # KNN 분류기 평가 (n_neighbors=5 사용)
    from sklearn.neighbors import KNeighborsClassifier
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    print("\n========== Evaluating KNeighborsClassifier ==========")
    evaluate_model(knn_clf, X_train, X_test, y_train, y_test, "KNeighborsClassifier", feature_names)
    
    # SVM 분류기 평가 (확률 예측을 위해 probability=True를 설정)
    from sklearn.svm import SVC
    svm_clf = SVC(probability=True, kernel='rbf')
    print("\n========== Evaluating SVM (SVC) Classifier ==========")
    evaluate_model(svm_clf, X_train, X_test, y_train, y_test, "SVM_Classifier", feature_names)
    
if __name__ == "__main__":
    main()