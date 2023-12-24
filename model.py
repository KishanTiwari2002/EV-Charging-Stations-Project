import pickle

# Load your dataset
# Replace 'your_dataset.csv' with the actual filename or path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

data1=pd.read_csv("C:\\Users\\prakash\\Downloads\\esv_data.csv")

data = pd.DataFrame(StandardScaler().fit_transform(data1), columns=data1.columns, index=data1.index)


x=data.drop(["Units","Load"],axis="columns")
y_load=data.iloc[:,-1]
y_units=data.iloc[:,-2]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y_load,test_size=0.3,random_state=0)
x1_train,x1_test,y1_train,y1_test=train_test_split(x,y_units,test_size=0.3,random_state=0)
load_model = RandomForestRegressor(random_state=42)
load_model.fit(x_train, y_train)
units_model = RandomForestRegressor(random_state=42)
units_model.fit(x1_train, y1_train)

import pickle

with open('load_model.pkl', 'wb') as f:
        pickle.dump(load_model, f)

# Save units_model to a pickle file
with open('units_model.pkl', 'wb') as f:
        pickle.dump(units_model, f)




