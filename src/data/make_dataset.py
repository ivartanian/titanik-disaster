import pandas as pd
from numpy import float32
from sklearn import tree

pd.set_option('expand_frame_repr', False)
pd.set_option('colheader_justify', 'center')

train = pd.read_csv("train.csv")
print("-------------------------HEAD----------------------------------")
print(train.head())
print("-------------------------SHAPE----------------------------------")
print(train.shape)
print("-------------------------COLUMNS----------------------------------")
print(train.columns)
print("-------------------------COLUMN NAMES, TYPES AND INFO----------------------------------")
print(train.dtypes)
print(train.info())
print("-------------------------PASSENGERS THAT SURVIVED VS PASSENGERS THAT PASSED AWAY AND PERCENTAGE----------------------------------")
survived_num = train["Survived"].value_counts()
print(survived_num)
survived_percentage = train.Survived.value_counts(normalize=True) * 100
print(survived_percentage)
print("-------------------------MALES THAT SURVIVED VS MALES THAT PASSED AWAY----------------------------------")
male_survived = train["Survived"][train["Sex"] == 'male'].value_counts()
print(male_survived)
print("-------------------------PRINT NORMALIZED SURVIVAL RATES FOR PASSENGERS UNDER 18----------------------------------")
passangers_children = train["Survived"][train["Age"] < 18].value_counts()
print(passangers_children)
passangers_adults = train["Survived"][train["Age"] >= 18].value_counts()
print(passangers_adults)
print("-------------------------DROP NA VALUES----------------------------------")
train = train.dropna( subset = ["Pclass", "Sex", "Age", "Fare", "Survived"], how = 'any')
print("-------------------------CONVERT THE MALE AND FEMALE GROUPS TO INTEGER FORM----------------------------------")
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
print(train.head())
print("-------------------------IMPUTE THE EMBARKED VARIABLE AND CONVERT TO INTEGER FORM----------------------------------")
train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
print(train.head())

pclass_num = train["Pclass"].value_counts()
print(pclass_num)
print("-------------------------DATA INFO----------------------------------")
print(train.info())
print("-------------------------CHANGE DATA TYPE----------------------------------")
train['PassengerId'] = train.PassengerId.astype(float32)
train['Survived'] = train.Survived.astype(float32)
train['Pclass'] = train.Pclass.astype(float32)
train['Sex'] = train.Sex.astype(float32)
train['Age'] = train.Age.astype(float32)
train['SibSp'] = train.SibSp.astype(float32)
train['Parch'] = train.Parch.astype(float32)
train['Fare'] = train.Fare.astype(float32)
print(train.info())
print("-------------------------CREATE THE TARGET AND FEATURES NUMPY ARRAYS: TARGET, FEATURES_ONE----------------------------------")
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one,target)
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one,target))