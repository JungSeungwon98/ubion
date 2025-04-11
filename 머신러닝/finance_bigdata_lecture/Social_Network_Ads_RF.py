import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df  = pd.read_csv('data/Social_Network_Ads.csv', encoding='utf-8',index_col=0)
# df = pd.DataFrame(data=data, columns=iris.feature_names)
# df['target'] = iris.target
# 1. LabelEncoder를 이용한 숫자형 카테고리 변환
label_encoder = LabelEncoder()
df['Gender']= label_encoder.fit_transform(df['Gender'])

print(df)

X = df.drop('Purchased' ,axis=1)
y = df['Purchased']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

# Print the evaluation indicators
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("AUC-ROC:", auc)

# Plot the ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# 새로운 데이터 (남자: 1, 나이: 72, 연봉: 35000)
new_data = pd.DataFrame({
    'Gender': [1],
    'Age': [72],
    'EstimatedSalary': [35000]
})

# 예측
prediction = rf.predict(new_data)
prediction_proba = rf.predict_proba(new_data)

print("구매 예측 결과 (0: 구매 안함, 1: 구매함):", prediction[0])
print("구매 확률 [클래스 0, 클래스 1]:", prediction_proba[0])