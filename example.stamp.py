# %%
from IPython.display import display
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier, plot_importance
# from lightgbm import LGBMClassifier
from mlxtend.plotting import plot_decision_regions
import tensorflow as tf


# %%
file_name = "data/20230907_All_Type_Vector_Avg.csv"
data = pd.read_csv(file_name)
data.head()


# %%
# 擷取資料集
X = data.drop(columns=["type", "approve"])
# y = data["approve"]
y = data["type"]
feature_names = X.columns.to_list()
target_names = y.value_counts().index.to_list()

print(feature_names)
print(target_names)
display(X.head())
display(y.head())


# %%
display(data["type"].value_counts())
display(data["approve"].value_counts())


# %%
plt.figure(figsize=(10, 10))
plt.title("Type Distribution")
plt.pie(data["type"].value_counts(), 
        labels=data["type"].value_counts().index, 
        labeldistance=1.05,
        autopct="%1.2f%%", 
        startangle=140)
plt.axis("equal")
plt.show()


# %%
plt.figure(figsize=(10, 10))
plt.title("Type Distribution")
plt.pie(data["approve"].value_counts(), 
        labels=data["approve"].value_counts().index, 
        labeldistance=1.05,
        autopct="%1.2f%%", 
        startangle=140)
plt.axis("equal")
plt.show()


# %%
# 標籤轉換
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)
y


# %%
# 預先分抽出測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
unique, counts = np.unique(y_test, return_counts=True)
dict(zip(unique, counts))


# %%
full_df = X_train.copy()
full_df["y"] = y_train
full_df


# %%
# 計算相關係數
corr_df = full_df.astype(float).corr()
corr_df


# %%
# 查詢出與y相關係數最高的欄位
corr_df["y"].sort_values(ascending=False)


# %%
# 分訓練集 & 驗證集
# 建立決策樹模型
model = DecisionTreeClassifier()
model = model.fit(X_train, y_train)


# %%
# Plot feature importances
feature_importances = pd.DataFrame(data=model.feature_importances_,
                                  columns=["important_feature"],
                                  index=feature_names)
feature_importances = feature_importances.sort_values(by="important_feature",
                                                    ascending=False)

feature_importances = feature_importances.head(10)

plt.figure(figsize=(16, 9))
plt.title("feature importances")
plt.xlabel("Feature")
plt.ylabel("Importance")
cmap = sns.cm.crest
sns.barplot(
    x=feature_importances.index,
    y=feature_importances["important_feature"],
    palette=cmap(feature_importances["important_feature"])
)
plt.show()


# %%
# Plot the decision tree
fig, ax = plt.subplots(figsize=(9, 16))
plot_tree(model,
          feature_names=feature_names,
          class_names=target_names,
          filled=True,
          proportion=True,
          rounded=True,
          ax=ax)
plt.show()


# %%
# 建立隨機森林模型
model = RandomForestClassifier()
model = model.fit(X_train, y_train)


# %%
# Xgboost
model = XGBClassifier()
model = model.fit(X_train, y_train)


# %%
# SVM
model = SVC()
model = model.fit(X_train, y_train)



# %%
# Predict
y_pred = model.predict(X_test)
pred_df = pd.DataFrame()
pred_df["預測結果"] = y_pred.tolist()
pred_df["實際結果"] = y_test.tolist()
pred_df["是否正確"] = pred_df["預測結果"] == pred_df["實際結果"]
pred_df


# %%
# Confusion Matrix
confusion_matrix_df = pd.DataFrame(data=confusion_matrix(y_test, y_pred),
                                   columns=target_names,
                                   index=target_names)
confusion_matrix_df


# %%
print(classification_report(y_test, y_pred,
                            zero_division=0))



# %%
# PCA降成2D
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_train_pca


# %%
# 查看PCA的主成分（特徵向量）
print("Principal Components (Eigenvectors):")
print(pca.components_)

# 查看PCA的主成分的方差解釋比例
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)


# %%
pca_df = pd.DataFrame({"PC1": X_train_pca[:, 0], "PC2": X_train_pca[:, 1], "Label": y_train})
pca_df


# %%
colors = ["red", "green", "black", "grey", "blue"]
labels = y_encoder.classes_

for i in range(len(labels)):
    subset = pca_df[pca_df["Label"] == i]
    plt.scatter(subset["PC1"], subset["PC2"], c=colors[i], label=labels[i])


plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA")
plt.legend()


plt.show()


# %%
# Tuning SVM
# Best Parameters:  {'C': 10, 'degree': 3, 'gamma': 0.1, 'kernel': 'rbf'}
param_grid = {'C': [0.1, 1, 10],
              'kernel': ['linear', 'rbf'],
              'gamma': [0.1, 1, 'auto'],
              'degree': [3, 4, 5]}

model = SVC()

grid_search = GridSearchCV(estimator=model, 
                           param_grid=param_grid, 
                           cv=5,
                           verbose=1,
                           n_jobs=-1)
grid_search.fit(X_train, y_train)


print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

best_estimator = grid_search.best_estimator_
y_pred = best_estimator.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy: ", accuracy)


# %%
# Tunning Xgboost
# Best Parameters: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.9}
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
}


model = XGBClassifier()

grid_search = GridSearchCV(estimator=model, 
                           param_grid=param_grid, 
                           cv=5,
                           verbose=1,
                           n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

best_estimator = grid_search.best_estimator_
y_pred = best_estimator.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy: ", accuracy)



# %%
y_test

# %%
# DNN
y_train = tf.keras.utils.to_categorical(y_train, 5)
y_test = tf.keras.utils.to_categorical(y_test, 5) 

# %%

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(768,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # 3 output classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=2)



# %%
# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")


# %%
# Predict
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
pred_df = pd.DataFrame()
pred_df["預測結果"] = y_pred_classes.tolist()
pred_df["實際結果"] = y_true_classes.tolist()
pred_df["是否正確"] = pred_df["預測結果"] == pred_df["實際結果"]
pred_df


# %%
# Confusion Matrix
confusion_matrix_df = pd.DataFrame(data=confusion_matrix(y_pred_classes, y_true_classes),
                                   columns=target_names,
                                   index=target_names)
confusion_matrix_df


# %%
print(classification_report(y_pred_classes, y_true_classes,
                            zero_division=0))


# %%
