from keras.utils.np_utils import to_categorical
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import numpy as np



def prepare_test_data(path_to_data):
    
    dataset = pd.read_csv("test.csv")
    X = get_features(dataset, [1,3,4,5,6,8,10], [0,2,3,4,5])

    return X

def prepare_train_data(path_to_data):
    dataset = pd.read_csv("train.csv")
    
    X = get_features(dataset, [2,4,5,6,7,9,11], [0,2,3,4,5])
    y = dataset.iloc[:,1].values
    y = to_categorical(y)

    return X, y    

def get_features(dataset, features_list, features_to_normalize):

    #Handling missing values
    dataset.iloc[:, -1].fillna('Z', inplace=True)
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())
    dataset['SibSp'] = dataset['SibSp'].fillna(0)
    dataset['Parch'] = dataset['Parch'].fillna(0)
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())


    X = dataset.iloc[: , features_list].values
    #X_view = dataset.iloc[: , features_list]

    
    #Normalize numerical features.
    min_max_scaler = MinMaxScaler()
        
    for i in features_to_normalize:
        min_max_scaler.fit(X[:,i].reshape(-1, 1))
        a= min_max_scaler.transform(X[:,i].reshape(-1, 1))
        X[:,i] = a[:,0] 


    #Transform categorical features into dummy varibles.
    le = LabelEncoder()
    X[:,1] = le.fit_transform(X[:,1])
    X[:,-1] = le.fit_transform(X[:,-1])
    
    
    one_hot_encoder = OneHotEncoder(sparse = False)
    
    features = one_hot_encoder.fit_transform(X[:,-1].reshape(-1,1))
    X = np.hstack((X[:,0:-1], features[:,[0,1,2]]))
    titles  = get_titles(dataset)
    np.array(titles).shape
    X = np.hstack((X[:,:], titles[:,:]) )

    return X

def titles_filter(title):
    accepted_titles =["Mr.", "Mrs.", "Miss.", "Master."]
    if title in accepted_titles:
        return title
    else:
        return "other"
    

def get_titles(dataset):
    le = LabelEncoder()
    all_titles =  map( lambda a : a.split(",")[1].split(" ")[1], dataset["Name"])
    filtred_titles = map(titles_filter, all_titles)
    num_categories = le.fit_transform(filtred_titles)
    one_hot_encoder = OneHotEncoder(sparse = False)
    dummy_varibles = one_hot_encoder.fit_transform(num_categories.reshape(-1,1))
    return dummy_varibles



    