import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading the titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")

# quick look before cleaning
print(df.head())
print(df.isnull().sum())

# --- cleaning up missing values ---
# age has a bunch of NaNs, filling with median since it's more robust to outliers
df["Age"] = df["Age"].fillna(df["Age"].median())

# only a couple missing in Embarked, just use the most common port
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# cabin is mostly empty anyway, not worth trying to fill it
df.drop(columns=["Cabin"], inplace=True)

sns.set_style("whitegrid")  # looks nicer than the default

# 1) how many survived vs didn't
plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.savefig("survival_count.png")
plt.show()

# 2) does gender affect survival chances?
plt.figure(figsize=(6, 4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Gender vs Survival")
plt.savefig("gender_survival.png")
plt.show()

# 3) age spread of passengers
plt.figure(figsize=(6, 4))
plt.hist(df["Age"], bins=20, color="steelblue", edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("age_distribution.png")
plt.show()

# 4) correlation between the numeric columns
plt.figure(figsize=(8, 6))
corr = df.select_dtypes(include="number").corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.show()

print("done - all plots saved!")