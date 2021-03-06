import pandas as pd
from numpy import float32
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

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
print(train.describe())
print("-------------------------MISSING DATA - AGE,EMBARKED----------------------------------")

train.pivot_table('PassengerId', 'Pclass', 'Survived', 'count').plot(kind='bar', stacked=True)

train['Age'] = train['Age'].fillna(train['Age'].median())
train["Embarked"] = train["Embarked"].fillna("S")
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
print("-------------------------CROSS TABULATION BETWEEN GENDER AND SURVIVED----------------------------------")
print(pd.crosstab(train.Sex, train.Survived, margins=True))
print(pd.crosstab(train.Sex, train.Survived, margins=True, normalize="index"))
print(pd.crosstab(train.Sex, train.Survived, margins=True, normalize="columns"))
print(pd.crosstab(train.Sex, train.Survived, margins=True, normalize="all"))
print(pd.crosstab([train.Embarked, train.Sex], train.Survived, margins=True))
print(pd.crosstab([train.Pclass, train.Sex], train.Survived, normalize="index"))
print("-------------------------PREPROCESSING DATA----------------------------------")
MaxPassEmbarked = train.groupby('Embarked').count()['PassengerId']
train.Embarked[train.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]

label = LabelEncoder()
dicts = {}

label.fit(train.Sex.drop_duplicates())
dicts['Sex'] = list(label.classes_)
train.Sex = label.transform(train.Sex)

# label.fit(train.Embarked.drop_duplicates())
# dicts['Embarked'] = list(label.classes_)
# train.Embarked = label.transform(train.Embarked)
print("-------------------------CONVERT THE MALE AND FEMALE GROUPS TO INTEGER FORM----------------------------------")
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
print(train.head())
print("-------------------------EMBARKED CONVERT TO INTEGER FORM----------------------------------")
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
print("-------------------------DECISION TREE----------------------------------")
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one.fit(train[predictors],train["Survived"])
scores = model_selection.cross_val_score(my_tree_one, train[predictors], train["Survived"], cv=model_selection.KFold(train.shape[0], random_state=1))
print("Accuracy and the 95% confidence interval of the estimate are: {0:.3f} (+/- {0:.2f})".format(scores.mean(), scores.std() * 2))
print("-------------------------LOGISTIC REGRESSION----------------------------------")
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
logreg = LogisticRegression(random_state=1)
scores = model_selection.cross_val_score(logreg, train[predictors], train["Survived"], cv=3)
print("Accuracy and the 95% confidence interval of the estimate are: {0:.3f} (+/- {0:.2f})".format(scores.mean(), scores.std() * 2))
print("-------------------------RANDOM FOREST----------------------------------")
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 25, random_state = 1)
forest.fit( train[predictors], train["Survived"])
feature_importances = forest.feature_importances_
scores = model_selection.cross_val_score(forest, train[predictors], train["Survived"], cv=model_selection.KFold(train.shape[0], random_state=1))
print("Accuracy and the 95% confidence interval of the estimate are: {0:.3f} (+/- {0:.2f})".format(scores.mean(), scores.std() * 2))