from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np

def logistic():
    #cloumns names
    column = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size',
              'Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",names=column)

    # print(data)
    #kill "?"
    data = data.replace(to_replace="?",value=np.nan)

    data = data.dropna()

    #Split data

    x_train,x_test,y_train,y_test = train_test_split(data[column[1:10]],data[column[10]],test_size=0.25)

    #Standard
    std = StandardScaler()

    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    lg = LogisticRegression(C=1.0)
    lg.fit(x_train,y_train)

    print(lg.coef_) #W value

    y_predict = lg.predict(x_test)
    print("Score:",lg.score(x_test,y_test))
    print("Call back:",classification_report(y_test,y_predict,labels=[2,4],target_names=["Good","Bad"])) #To see precision rate





if __name__ == '__main__':
    logistic()