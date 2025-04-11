import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import graphviz
from sklearn.tree import export_graphviz


#Social_Network_Ads.csv Load

df = pd.read_csv("data/Social_Network_Ads.csv", encoding='utf-8', index_col= 0)
# df = pd.DataFrame(data=data, columns=iris.feature_names)
# df['target'] = iris.target
# 1. LabelEncoder를 이용한 숫자형 카테고리 변환
label_encoder = LabelEncoder()
df['Gender']= label_encoder.fit_transform(df['Gender'])

print(df)

X = df.drop('Purchased', axis=1)
y = df['Purchased']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Create a dot language string to represent the decision tree
dot_data = export_graphviz(clf, out_file="tree.dot", feature_names=X.columns, class_names=[str(cls) for cls in clf.classes_], filled=True)

# Create a Graphviz object and render the graph
graphviz.Source(dot_data).render('decision_tree', format='png')
# Calculate feature importance
feature_importances = clf.feature_importances_

# Print the feature importances
print("Feature Importances:")
for i, feature in enumerate(X_train.columns):
    print(f"{feature}: {feature_importances[i]:.4f}")

# Make predictions on the testing set
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
# print(y_pred_proba)
# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
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
prediction = clf.predict(new_data)
prediction_proba = clf.predict_proba(new_data)

print("구매 예측 결과 (0: 구매 안함, 1: 구매함):", prediction[0])
print("구매 확률 [클래스 0, 클래스 1]:", prediction_proba[0])