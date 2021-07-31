from sklearn import preprocessing
import pandas as pd
housing = pd.read_csv("/content/sample_data/california_housing_train.csv")
d = preprocessing.normalize(housing, axis=0)
scaled_df = pd.DataFrame(d, columns=names)
scaled_df.head()
