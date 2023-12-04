import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def sort_dataset(dataset_df):
    sorted_dataset_df = dataset_df.sort_values(by='year', ascending=True)
    
    return sorted_dataset_df

def split_dataset(dataset_df):
    dataset_df['salary'] = dataset_df['salary'] * 0.001

    past_data = dataset_df[dataset_df['year'] < 2018]
    future_data = dataset_df[dataset_df['year'] >= 2018]
		
    X_train = past_data.drop(columns='salary')
    Y_train = past_data['salary']
    X_test = future_data.drop(columns='salary')
    Y_test = future_data['salary']
		
    return X_train, X_test, Y_train, Y_test
	
def extract_numerical_cols(dataset_df):
    columns = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']
    numerical_columns = dataset_df[columns]

    return numerical_columns

def train_predict_decision_tree(X_train, Y_train, X_test):
    dt_reg = DecisionTreeRegressor()
    dt_reg.fit(X_train, Y_train)
    predicted = dt_reg.predict(X_test)
    
    return predicted
	
def train_predict_random_forest(X_train, Y_train, X_test):
    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_train, Y_train)
    predicted = rf_reg.predict(X_test)
    
    return predicted

def train_predict_svm(X_train, Y_train, X_test):
    svm_pipe = make_pipeline(
        StandardScaler(),
        SVR()
    )
    svm_pipe.fit(X_train, Y_train)
    predicted = svm_pipe.predict(X_test)
    
    return predicted

def calculate_RMSE(labels, predictions):
		RMSE = np.sqrt(np.mean((predictions-labels)**2))
                
		return RMSE

if __name__=='__main__':
    #DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
    sorted_df = sort_dataset(data_df)	
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
    print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
    print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
    print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))