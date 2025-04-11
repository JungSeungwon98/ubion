# -*- coding: utf-8 -*-
"""
예제: KNN 알고리즘을 활용하여 첨부한 Social_Network_Ads_outlier_removed.csv 데이터로 예측하고,
이전 코드에서 사용한 평가지표(정확도, 혼동 행렬, 분류 보고서, 정밀도, 재현율, F1-score, 
교차 검증, ROC 커브, 러닝 커브, (모델이 지원하는 경우) 특성 중요도)를 모두 반영한 코드입니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize, LabelEncoder

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, feature_names):
    # 1. 모델 학습
    model.fit(X_train, y_train)
    
    # 2. 예측 수행
    y_pred = model.predict(X_test)
    
    # 3. Accuracy (정확도)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.3f}")
    
    # 4. Confusion Matrix (혼동 행렬)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{model_name} Confusion Matrix:\n{cm}")
    
    # 5. Classification Report (분류 보고서)
    report = classification_report(y_test, y_pred)
    print(f"\n{model_name} Classification Report:\n{report}")
    
    # 6. Precision, Recall, F1-score (macro 평균)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"{model_name} Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")
    
    # 7. Cross Validation Scores (5-겹 교차 검증)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{model_name} 5-Fold Cross Validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # 8. ROC AUC 및 ROC Curve (다중 클래스의 경우 one-vs-rest 방식)
    # 정답 레이블 이진화: 이진 분류의 경우 label_binarize는 (n_samples, 1) 배열을 반환하므로 확장 필요
    classes = np.unique(y_train)
    y_test_bin = label_binarize(y_test, classes=classes)
    if y_test_bin.shape[1] == 1:  # 이진 분류인 경우 두 열로 확장
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
    
    # 9. Learning Curve 그리기
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Cross-validation score")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.title(f'{model_name} Learning Curve')
    plt.legend(loc="best")
    plt.show()
    
    # 10. Feature Importances (특성 중요도) - KNN은 지원하지 않으므로 건너뜁니다.
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
    df = pd.read_csv('data/Social_outlier_removed.csv', encoding='utf-8', index_col=0)
    
    # 필요한 경우, 범주형 변수(예: Gender)를 숫자로 변환
    label_encoder = LabelEncoder()
    if 'Gender' in df.columns:
        df['Gender'] = label_encoder.fit_transform(df['Gender'])
    
    print("데이터 미리보기:")
    print(df.head())
    
    # 특성과 타겟 분리 (일반적으로 'Purchased'가 타겟)
    X = df.drop('Purchased', axis=1)
    y = df['Purchased']
    feature_names = X.columns
    
    # 학습/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # KNN 분류기 생성 (기본 n_neighbors=5 사용)
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    
    print("\n========== Evaluating KNeighborsClassifier ==========")
    evaluate_model(knn_clf, X_train, X_test, y_train, y_test, "KNeighborsClassifier", feature_names)

if __name__ == "__main__":
    main()