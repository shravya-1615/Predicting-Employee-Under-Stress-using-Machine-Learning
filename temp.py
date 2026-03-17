import pandas as pd
df= pd.read_csv('./emp_train_new.csv')
df.dropna(inplace=True)
df.to_csv('./newdf.csv',sep=',')

print(len(df))