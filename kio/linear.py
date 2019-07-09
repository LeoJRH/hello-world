from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression,SGDRegressor, Ridge
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

def mylin():
    """"
    Linear predict
    """

    #Data
    lb = load_boston()


    #Split data

    x_train,x_test,y_train,y_test = train_test_split(lb.data, lb.target,test_size=0.25)

  #  print(y_train,y_test)

    #Standardizing data and target
    std_x = StandardScaler()

    x_train = std_x.fit_transform(x_train)
    x_test = std_x.fit_transform(x_test)

    std_y = StandardScaler()

    y_train = std_y.fit_transform(y_train.reshape(-1,1)) #parameter two-dimension
    y_test = std_y.fit_transform(y_test.reshape(-1,1))
    lr = joblib.load("./test.pkl") #
    y_predict = std_y.inverse_transform(lr.predict(x_test))

    print("Data from pkl:",y_predict)

    # lr = LinearRegression()
    # lr.fit(x_train,y_train)
    # print(lr.coef_) #value W
    #
    # #Save model
    # joblib.dump(lr,"./test.pkl")
    # y_lr_predict = std_y.inverse_transform(lr.predict(x_test)) #set value back without standardizing
    # print("Y_predict 's",y_lr_predict)
    #
    # print("Lr rate:", mean_squared_error(std_y.inverse_transform(y_test),y_lr_predict))




    # # #SGD
    # sgd = SGDRegressor()
    #
    # sgd.fit(x_train,y_train)
    # print(sgd.coef_) #value W
    # y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test)) #set value back without standardizing
    # print("Y_predict 's",y_sgd_predict)
    # print("Sgd rate:", mean_squared_error(std_y.inverse_transform(y_test),y_sgd_predict))
    # #Ridge
    #
    # rd = Ridge(alpha=1.0)
    #
    # rd.fit(x_train,y_train)
    # print(sgd.coef_) #value W
    # y_rd_predict = std_y.inverse_transform(rd.predict(x_test)) #set value back without standardizing
    # print("Y_predict 's",y_rd_predict)
    # print("Ridge rate:", mean_squared_error(std_y.inverse_transform(y_test),y_rd_predict))


if __name__ == '__main__':
    mylin()