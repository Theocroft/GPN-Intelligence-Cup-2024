## Импорт стандартных библиотек
import pandas as pd
import numpy as np

## Прочие библиотеки
import warnings
warnings.filterwarnings("ignore")

# Метрики
from sklearn import model_selection, metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#Выбор моделей
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

# Модели машинного обучения
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

# Ансамбли
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor

# Прочее 
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Data/Processed/Final.csv")
df = pd.DataFrame(df)

X = df.drop("Гармония Бессмертия", axis = 1 )
Y = df['Гармония Бессмертия']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=42)
st = StandardScaler()
X_train = st.fit_transform(X_train)
X_test = st.transform(X_test)


model_LR = LinearRegression()
model_LR.fit(X_train, y_train)  

model_Ridge = Ridge()
model_Ridge.fit(X_train, y_train)  

model_Lasso = Lasso()
model_Lasso.fit(X_train, y_train)  

ensemble_stack = StackingRegressor([('lr', LinearRegression()),
                            ('Ridge', Ridge()),
                            ('Lasso', Lasso())])
ensemble_stack.fit(X_train, y_train)  


import pickle
LR_Name = "LR Model_Final.pkl"  

with open(LR_Name, 'wb') as file:  
    pickle.dump(model_LR, file)

Ridge_Name = "Ridge Model_Final.pkl"  

with open(Ridge_Name, 'wb') as file:  
    pickle.dump(model_Ridge, file)

Lasso_Name = "Lasso Model_Final.pkl"  

with open(Lasso_Name, 'wb') as file:  
    pickle.dump(model_Lasso, file)
    
Stack_Name = "Stacking Model_Final.pkl"  

with open(Stack_Name, 'wb') as file:  
    pickle.dump(ensemble_stack, file)

