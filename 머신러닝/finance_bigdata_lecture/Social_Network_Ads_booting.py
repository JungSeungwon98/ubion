# -*- coding: utf-8 -*-
"""
예제: sklearn의 boosting 기법을 활용한 분류 모델 평가 코드
AdaBoostClassifier와 GradientBoostingClassifier를 사용하고,
모델 성능을 평가하기 위한 다양한 지표(정확도, 혼동 행렬, 분류 보고서, ROC 커브, 러닝 커브, 특성 중요도 등)를 포함합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    
    # 6. ROC AUC 및 ROC Curve (다중 클래스: one-vs-rest 방식)
    # 다중 클래스인 경우를 위해 정답 레이블 이진화
    classes = np.unique(y_train)
    y_test_bin = label_binarize(y_test, classes=classes)
    # 이진 분류인 경우, y_test_bin의 shape는 (n_samples,1)이므로 두 개의 열로 확장
    if y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))
    
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
        try:
            roc_auc = roc_auc_score(y_test_bin, y_score, multi_class='ovr', average='macro')
            print(f"{model_name} ROC AUC (macro, one-vs-rest): {roc_auc:.3f}")
        except ValueError as e:
            print(f"ROC AUC 계산 중 오류: {e}")
        
        # ROC Curve 그리기 (각 클래스별)
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
    
    # 8. Feature Importances (특성 중요도) - 모델이 지원하는 경우
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
    # Social_Network_Ads.csv 파일 로드
    df = pd.read_csv('data/Social_Network_Ads.csv', encoding='utf-8', index_col=0)
    # 1. LabelEncoder를 이용한 숫자형 카테고리 변환
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    
    print(df)
    
    X = df.drop('Purchased', axis=1)
    y = df['Purchased']
    feature_names = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # AdaBoost 분류기 (약한 학습기로 결정 트리 사용)
    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
    ada_clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)
    print("========== Evaluating AdaBoostClassifier ==========")
    evaluate_model(ada_clf, X_train, X_test, y_train, y_test, "AdaBoostClassifier", feature_names)
    
    # Gradient Boosting 분류기
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    print("\n========== Evaluating GradientBoostingClassifier ==========")
    evaluate_model(gb_clf, X_train, X_test, y_train, y_test, "GradientBoostingClassifier", feature_names)
    
if __name__ == "__main__":
    main()