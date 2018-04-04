import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('colheader_justify', 'center')

train = pd.read_csv("train.csv")
print("-------------------------HEAD----------------------------------")
print(train.head())
print("-------------------------SHAPE----------------------------------")
print(train.shape)
print("-------------------------COLUMNS----------------------------------")
print(train.columns)
print("-------------------------COLUMN NAMES AND TYPES----------------------------------")
print(train.dtypes)
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
print("-------------------------CONVERT THE MALE AND FEMALE GROUPS TO INTEGER FORM----------------------------------")
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
print(train.head())