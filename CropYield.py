from tkinter import *
from tkinter.filedialog import askopenfilename
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge



main = tkinter.Tk()
main.title("CROP YIELD PREDICTION AND EFFICIENT USE OF FERTILIZERS USING MACHINE LEARNING")
main.geometry("1300x1200")


global filename
global X_train, X_test, y_train, y_test
global X,Y
global dataset
global le
global model
global fertilizer

def upload():
    global filename
    global dataset
    global fertilizer
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,'Tweets dataset loaded\n')
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    dataset['Production'] = dataset['Production'].astype(np.int64)
    fertilizer = pd.read_csv("Dataset/Fertilizer_Prediction.csv")
    fertilizer.fillna(0, inplace = True)
    fertilizer = fertilizer.values
    text.insert(END,str(dataset.head())+"\n")

def processDataset():
    global le
    global dataset
    global X_train, X_test, y_train, y_test
    global X,Y
    text.delete('1.0', END)
    le = LabelEncoder()

    dataset['State_Name'] = pd.Series(le.fit_transform(dataset['State_Name']))
    dataset['District_Name'] = pd.Series(le.fit_transform(dataset['District_Name']))
    dataset['Season'] = pd.Series(le.fit_transform(dataset['Season']))
    dataset['Crop'] = pd.Series(le.fit_transform(dataset['Crop']))
    text.insert(END,str(dataset.head())+"\n")
    datasets = dataset.values
    cols = datasets.shape[1]-1
    X = datasets[:,0:cols]
    Y = datasets[:,cols]
    Y = Y.astype('uint8')
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END,"\n\nTotal records found in dataset is : "+str(len(X))+"\n")
    text.insert(END,"80% records used to train machine learning algorithm : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% records used to train machine learning algorithm : "+str(X_test.shape[0])+"\n")


def trainModel():
    global model
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    global X,Y

    model = DecisionTreeRegressor(max_depth=100,random_state=0,max_leaf_nodes=20,max_features=5,splitter="random")
    model.fit(X,Y)
    predict = model.predict(X_test)
    mse = mean_squared_error(predict,y_test)
    rmse = np.sqrt(mse)/ 1000
    text.insert(END,"Training process completed\n")
    text.insert(END,"Decision Tree Machine Learning Algorithm Training RMSE Error Rate : "+str(rmse)+"\n\n")

def LinearModel():
    global model1
    #text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    global X,Y

    model1 = LinearRegression()
    model1.fit(X,Y)
    predict = model1.predict(X_test)
    mse1 = mean_squared_error(predict,y_test)
    rmse1 = np.sqrt(mse1)/ 1000
    text.insert(END,"Training process completed\n")
    text.insert(END,"Linear Regression Machine Learning Algorithm Training RMSE Error Rate : "+str(rmse1)+"\n\n")

"""
def LogisticModel():
    global model2
    #text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    global X,Y

    model2 = Ridge()
    model2.fit(X,Y)
    predict = model2.predict(X_test)
    mse2 = mean_squared_error(predict,y_test)
    rmse2 = np.sqrt(mse2)/ 1000
    text.insert(END,"Training process completed\n")
    text.insert(END,"Ridge Regression Machine Learning Algorithm Training RMSE Error Rate : "+str(rmse2)+"\n\n")

"""
  
    
def getFertilizer(crop):
    global fertilizer
    crop = crop.strip().lower()
    fert = 'none'
    for i in range(len(fertilizer)):
        name = fertilizer[i,4]
        fert_name = fertilizer[i,8]
        name = name.strip().lower()
        if name == crop:
            fert = fert_name
            print("got original "+fert)
            break
    if fert == 'none':
        fertilizers = pd.read_csv("Dataset/Fertilizer_Prediction.csv")
        fert_name = fertilizers['Fertilizer_Name']
        fert_name = np.asarray(fert_name)
        fert = fert_name[random.randint(0,95)]
        print("got random "+fert)
    return fert        

def cropYieldPredict():
    global model
    global le
    text.delete('1.0', END)
    testname = filedialog.askopenfilename(initialdir = "Dataset")
    test = pd.read_csv(testname)
    names = pd.read_csv(testname)
    names = names['Crop']
    test.fillna(0, inplace = True)
    test['State_Name'] = pd.Series(le.fit_transform(test['State_Name']))
    test['District_Name'] = pd.Series(le.fit_transform(test['District_Name']))
    test['Season'] = pd.Series(le.fit_transform(test['Season']))
    test['Crop'] = pd.Series(le.fit_transform(test['Crop']))
    test = test.values
    test = normalize(test)
    cols = test.shape[1]
    test = test[:,0:cols]
    predict = model.predict(test)
    for i in range(len(predict)):
        production = predict[i] * 100
        crop_yield = (production / 10000) / 10
        fert_name = getFertilizer(names[i])
        text.insert(END,"Test Record : "+str(test[i])+"\nProduction would be : "+str(production)+" KGs\n")
        text.insert(END,"Yield would be : "+str(crop_yield)+" KGs/acre\n")
        text.insert(END,"Recommended Fertilizer : "+str(fert_name)+"\n\n")
        
def plotComparison():
    global model, model1 #, model2
    global X_test, y_test 

    predict_dt = model.predict(X_test)
    predict_lr = model1.predict(X_test)
    #predict_rr = model2.predict(X_test)

    mse_dt = mean_squared_error(predict_dt, y_test)
    rmse_dt = np.sqrt(mse_dt) / 1000

    mse_lr = mean_squared_error(predict_lr, y_test)
    rmse_lr = np.sqrt(mse_lr) / 1000

    #mse_rr = mean_squared_error(predict_rr, y_test)
    #rmse_rr = np.sqrt(mse_rr) / 1000

    models = ['Decision Tree', 'Linear Regression']#, 'Ridge Regression']
    rmse_values = [rmse_dt, rmse_lr] #, rmse_rr]

    plt.bar(models, rmse_values, color=['blue', 'green'])#, 'red'])
    plt.xlabel('Machine Learning Algorithm')
    plt.ylabel('RMSE (in thousands)')
    plt.title('Comparison of RMSE Among Algorithms')
    plt.show()
    

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='crop yield prediction and efficient use of fertilizers using machine learning')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Crop Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
#pathlabel.place(x=700,y=100)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=700,y=150)
processButton.config(font=font1) 

mlButton = Button(main, text="Decision Tree Machine Learning Algorithm", command=trainModel)
mlButton.place(x=700,y=200)
mlButton.config(font=font1)

mlButton = Button(main, text="Linear Regression Algorithm", command=LinearModel)
mlButton.place(x=700,y=250)
mlButton.config(font=font1)
'''
mlButton = Button(main, text="Ridge Regression Algorithm", command=LogisticModel)
mlButton.place(x=700,y=300)
mlButton.config(font=font1)
'''
closeButton = Button(main, text="Comparison of RMSE among Algorithm ", command=plotComparison)
closeButton.place(x=700,y=300)
closeButton.config(font=font1)

predictButton = Button(main, text="Upload Test Data & Predict Yield", command=cropYieldPredict)
predictButton.place(x=700,y=350)
predictButton.config(font=font1)


closeButton = Button(main, text="Close", command=close)
closeButton.place(x=700,y=400)
closeButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
