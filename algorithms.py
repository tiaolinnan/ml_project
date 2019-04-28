import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import r2_score






# reading the dataset
df = pd.read_csv("train_data.csv", sep=",")

# it may be needed in the future.
serialNo = df["Serial No."].values

df.drop(["Serial No."], axis=1, inplace=True)

df = df.rename(columns={'Chance of Admit ': 'Chance of Admit'})

y = df["Chance of Admit"].values
x = df.drop(["Chance of Admit"], axis=1)

# separating train (80%) and test (%20) sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

scalerX = MinMaxScaler(feature_range=(0, 1))
x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = scalerX.transform(x_test[x_test.columns])

lr = LinearRegression()
lr.fit(x_train,y_train)
y_head_lr = lr.predict(x_test)

print "Linear Regression: "
print "real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(lr.predict(x_test.iloc[[1],:]))
print "real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(lr.predict(x_test.iloc[[2],:]))

from sklearn.metrics import r2_score
print "r_square score: ", r2_score(y_test,y_head_lr)

y_head_lr_train = lr.predict(x_train)
print "r_square score (train dataset): ", r2_score(y_train,y_head_lr_train)

print "----------------"
print "Decision Tree: "
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(x_train, y_train)
y_head_dtr = dtr.predict(x_test)

print "r_square score: ", r2_score(y_test,y_head_dtr)
print "real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(dtr.predict(x_test.iloc[[1],:]))
print "real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(dtr.predict(x_test.iloc[[2],:]))

y_head_dtr_train = dtr.predict(x_train)
print "r_square score (train dataset): ", r2_score(y_train,y_head_dtr_train)


print "----------------"
print "Random Forest:"

rfr = RandomForestRegressor(n_estimators = 100, random_state = 42)
rfr.fit(x_train, y_train)
y_head_rfr = rfr.predict(x_test)

print "r_square score: ", r2_score(y_test, y_head_rfr)
print "real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(rfr.predict(x_test.iloc[[1],:]))
print "real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(rfr.predict(x_test.iloc[[2],:]))


y_head_rf_train = rfr.predict(x_train)
print "r_square score (train dataset): ", r2_score(y_train,y_head_rf_train)


y = np.array([r2_score(y_test,y_head_lr),r2_score(y_test,y_head_rfr),r2_score(y_test,y_head_dtr)])
x = ["LinearRegression","RandomForestReg.","DecisionTreeReg."]
plt.bar(x,y)
plt.title("Comparison of Regression Algorithms")
plt.xlabel("Regressor")
plt.ylabel("r2_score")
plt.show()