import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from  data import prepare_test_data, prepare_train_data





def main():
    hidden_size = 50
    dropout_rate = 0.1
    X, X_test, y, y_test, competition_X_test = get_data()
    model = train_model(X, y, hidden_size = hidden_size, dropout_rate= dropout_rate)
    
    #Checking of model accuracy on prepared test data.
    y_pred = model.predict(X_test)
    cm = prepare_confusion_matrix(y_pred, y_test)
    print cm
    
    #Getting result predictions on competition test data and writing it to csv file.
    result = test_data_result(competition_X_test, model)
    result.to_csv("result.csv", index = False)
    
def get_data():
    competition_X_test = prepare_test_data("test.csv")
    X, y = prepare_train_data("train.csv")
    X, X_test, y, y_test = train_test_split(X, y, test_size = 0.1)
    return X, X_test, y, y_test, competition_X_test
    
def train_model(X, y, hidden_size = 50, dropout_rate = 0.1,
                nb_epoch = 5, batch_size = 20):
    
    model = prepare_network(14, hidden_size, dropout_rate)
    model.fit(X, y, nb_epoch =5, batch_size = 20)
    return model    

  
def prepare_network(features_num, hidden_size, drop_out):
    model = Sequential()
    model.add(Dense(hidden_size, activation='relu', input_dim=features_num))
    model.add(Dropout(drop_out))
    model.add(Dense(hidden_size, activation='relu', input_dim=features_num))
    model.add(Dropout(drop_out))
    model.add(Dense(2, activation='softmax'))
    
    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])    
    return model    

       
def test_data_result(X, network):    
    dataset = pd.read_csv("test.csv")
    results = network.predict(X)
    survived = []
    
    for result in results:
        survived.append(int(result[0]< result[1]))

    
    df2 = pd.DataFrame({"PassengerId":dataset["PassengerId"],
                        "Survived" : survived
                        })
    return df2

def prepare_confusion_matrix(y_pred, y_test):
    survived_pred = []
    for result in y_pred:
        survived_pred.append(int(result[0]< result[1]))
        
    survived_test = []
    for result in y_test:
        survived_test.append(int(result[0]< result[1]))
    return confusion_matrix(survived_pred, survived_test)   
    

    



if __name__=="__main__":
    main()

















