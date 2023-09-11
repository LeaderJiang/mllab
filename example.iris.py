# %%
# 安裝需要函式庫
!pip3 install numpy pandas matplotlib seaborn scikit-learn statsmodels mlxtend
# !pip3 install plotly
# !pip3 install ipykernel
# !pip3 install nbformat --upgrade


# %%
# 載入需要的函式庫
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions


# %%
# Load Dataset
dataset = datasets.load_iris()
display(dataset.data)
display(dataset.target)
display(dataset.feature_names)
target_names = dataset.target_names.tolist()
display(target_names)


# %%
df = pd.DataFrame(data=dataset["data"], 
                  columns=dataset["feature_names"])
df["target"] = dataset["target"]
df


# %%
# 看一下資料內容
print("===== df.info =====")
display(df.info())
print("===== df.describe =====")
# 描述資料
display(df.describe())


# %%
# 畫圖看資料分布趨勢
sns.pairplot(df)
plt.show()


# %%
# Correlation
plt.figure(figsize=(16, 9))
plt.title("Correlation of Features", y=1.01, size=15)
sns.heatmap(
    df.corr(method="pearson", numeric_only=True),
    square=True,
    linewidths=0.1,
    linecolor="white",
    annot=True,
)
plt.show()


# %%
# 我們把我們擁有的資料集分成兩份, 一份測試, 一份訓練 -> 82法則
X_train, X_test, y_train, y_test = train_test_split(df[["sepal length (cm)", 
                                                        "sepal width (cm)", 
                                                        "petal length (cm)", 
                                                        "petal width (cm)"]],
                                                    df["target"],
                                                    test_size=0.2)
X_train


# %%
# 建立決策樹模型
model = DecisionTreeClassifier(max_depth=3)
model = model.fit(X_train, y_train)


# %%
# Plot the decision tree
fig, ax = plt.subplots(figsize=(9, 16))
plot_tree(model,
          feature_names=dataset.feature_names,
          class_names=dataset.target_names.tolist(),
          filled=True,
          proportion=True,
          rounded=True,
          ax=ax)
plt.show()


# %%
# Cross Valid Score
max_depths = np.arange(1, 21)
average_scores = []

for depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(estimator=clf, 
                             X=X_train, 
                             y= y_train,
                             cv=3)
    average_scores.append(np.mean(scores))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(max_depths, average_scores, marker='o', linestyle='-')
plt.title('max_depth vs. accuracy')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.grid(True)
plt.show()


# %%
# 建立隨機森林模型
model = RandomForestClassifier()
model = model.fit(X_train, y_train)


# %%
# 決策樹判斷重要參數作法
important_features = pd.DataFrame(data=model.feature_importances_,
                                  columns=["important_feature"],
                                  index=dataset["feature_names"])
important_features = important_features.sort_values(by="important_feature",
                                                    ascending=False)

plt.figure(figsize=(16, 9))
plt.xlabel("Feature")
plt.ylabel("Importance")
cmap = sns.cm.crest
sns.barplot(
    x=important_features.index,
    y=important_features["important_feature"],
    palette=cmap(important_features["important_feature"]),
)
plt.show()

important_features


# %%
# Step 3. 開始預測
y_pred = model.predict(X_test)
pred_df = pd.DataFrame()
pred_df["預測結果"] = y_pred.tolist()
pred_df["實際結果"] = y_test.tolist()
pred_df["是否正確"] = pred_df["預測結果"] == pred_df["實際結果"]
pred_df


# %%
# Confusion Matrix
confusion_matrix_df = pd.DataFrame(data=confusion_matrix(y_test, y_pred),
                                   columns=dataset["target_names"],
                                   index=dataset["target_names"])
# x: predict, y: true
confusion_matrix_df


# %%
print(classification_report(y_test, y_pred, 
                            target_names=dataset["target_names"]))


# %%
# Decision Regions
dataset = datasets.load_iris()
X = dataset.data[:, :2]
y = dataset.target


classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    LogisticRegression(),
    SVC(),
]


fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, clf in zip(axes.ravel(), classifiers):
    # 配適分類器
    clf.fit(X, y)
    # 繪製決策邊界
    plot_decision_regions(X, y, clf=clf, legend=2, ax=ax)
    ax.set_xlabel(dataset.feature_names[0])
    ax.set_ylabel(dataset.feature_names[1])
    ax.set_title(clf.__class__.__name__)


plt.tight_layout()
plt.show()


