import pandas as pd

df=pd.read_csv("data/raw/clean.csv")
print(df.head(2))

# temp=df['Season'].value_counts()
# print(len(temp.index))
cat_col_list=["conditions","description","icon","source","City","Month","Season","Day_of_Week"]
for col in cat_col_list:
    curr_val=df[col].value_counts().index
    # if len(curr_val)==1:
    print(f"Column Name: {col}")
    print(curr_val)
    print(f"len: {len(curr_val)}")