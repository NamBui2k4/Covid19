
import pandas as pd
import numpy as np

df = pd.read_csv('corona_virus.csv',  encoding='unicode_escape')
df

print(df.columns)
print(df.info())
print(df.describe())




# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.svm import SVR

# # splitting data
# target = 'Deaths/1M pop'
# X = df.drop(target, axis=1)
# y = df[target]

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
# nom_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
# preprocessor    = ColumnTransformer(transformers=[
#     ('catagory_features', nom_transformer, ['Country,Other']),
#     ('numeric_features', num_transformer, [col for col in X.columns if col != 'Country,Other'] )
# ])

# reg = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('svm for regression', SVR())
# ])

# reg.fit(X_train, y_train)

# y_pred = reg.predict(X_test) # dự đoán không chính xác cho lắm :D

# for i, j in zip(y_test, y_pred):
#     print(f'Actual: {i}, Predicted: {j}')