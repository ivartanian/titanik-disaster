import pandas as pd
from numpy import float32
from sklearn import tree
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import figure_factory as FF

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
print("-------------------------ADD NEW COLUMN AND MAP VALUES OF AN EXISTING COLUMN----------------------------------")
train["gender"] = train.Sex.map( lambda x: int( x == 'male') )
print("-------------------------GROUPING AND AGGREGATING----------------------------------")
pclass_gender_age_mean_df = train.groupby( ['Pclass', 'Sex'] )['Age'].mean().reset_index()
print(pclass_gender_age_mean_df)
pclass_gender_survival_count_df = train.groupby( ['Pclass', 'Sex'] )['Survived'].sum().reset_index()
print(pclass_gender_survival_count_df)
pclass_age_gender_survival_df = pclass_gender_age_mean_df.merge(pclass_gender_survival_count_df, on = ['Pclass', 'Sex'])
print(pclass_age_gender_survival_df)
print("-------------------------WHO WAS MORE LIKELY TO SURVIVE FEMALE OR MALE?----------------------------------")
# Normalized male survival
male_survival = train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)
# Normalized female survival
female_survival = train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)

# Survival by Sex
x0 = ['male', 'female']
y0 = [male_survival[1], female_survival[1]]
data = [go.Bar(x=x0, y=y0)]
layout = go.Layout(autosize = False, width = 300, height = 400,
              yaxis = dict(title = 'Survival Rates'),
              title = 'Survival by Sex')
fig1 = go.Figure(data = data, layout = layout)
py.iplot(fig1)
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
print("-------------------------ONLY AGE, SEX AND PCLASS OF PASSENGERS WHO HAVE SURVIVED----------------------------------")
print(train[ ( train.Survived == 1 ) & ( train.Age <= 5 ) ][['Age', 'Sex', 'Pclass']][0:5])
print("-------------------------CREATE THE TARGET AND FEATURES NUMPY ARRAYS: TARGET, FEATURES_ONE----------------------------------")
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one,target)
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one,target))