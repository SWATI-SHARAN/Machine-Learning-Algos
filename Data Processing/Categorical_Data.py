import pandas as pd # pandas library, which is used for handling tabular data (like DataFrames).
data = {"Color": ["Red", "Blue", "Green", "Blue", "Red"],
        "Size": ["S", "M", "L", "M", "S"]}
df = pd.DataFrame(data)
df_encoded = pd.get_dummies(df, columns=["Color", "Size"], drop_first=True) # categorical variables into binary (0/1) dummy variables.
#drop_first=True avoids multicollinearity by dropping the first category from each feature.

print("Original Dataframe:", df)
print("\n Encoded Dataframe:", df_encoded)