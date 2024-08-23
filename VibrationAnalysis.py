# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:36:26 2023

@author: soere
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

from sklearn.metrics import classification_report, f1_score


# Define the file path
Data_0D_path = r'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 3\Machine Learning\Poster Project\0D.csv'
Data_1D_path = r"C:\Users\soere\OneDrive\UNI\Kandidat\Semester 3\Machine Learning\Poster Project\1D.csv"
Data_2D_path = r"C:\Users\soere\OneDrive\UNI\Kandidat\Semester 3\Machine Learning\Poster Project\2D.csv"
Data_3D_path = r"C:\Users\soere\OneDrive\UNI\Kandidat\Semester 3\Machine Learning\Poster Project\3D.csv"
Data_4D_path = r"C:\Users\soere\OneDrive\UNI\Kandidat\Semester 3\Machine Learning\Poster Project\4D.csv"


Data_0E_path = r"C:\Users\soere\OneDrive\UNI\Kandidat\Semester 3\Machine Learning\Poster Project\0E.csv"
Data_1E_path = r"C:\Users\soere\OneDrive\UNI\Kandidat\Semester 3\Machine Learning\Poster Project\1E.csv"
Data_2E_path = r"C:\Users\soere\OneDrive\UNI\Kandidat\Semester 3\Machine Learning\Poster Project\2E.csv"
Data_3E_path = r"C:\Users\soere\OneDrive\UNI\Kandidat\Semester 3\Machine Learning\Poster Project\3E.csv"
Data_4E_path = r"C:\Users\soere\OneDrive\UNI\Kandidat\Semester 3\Machine Learning\Poster Project\4E.csv"


UpperCut_D = 3100000 + 100000
UpperCut_E = 700000 + 100000
# Read the CSV file into DataFrame
Data_0D = pd.read_csv(Data_0D_path, nrows=UpperCut_D)

# 1D data

# Unbalanced moment of inertia for eccentric mass

Data_0D = pd.read_csv(Data_0D_path, nrows=UpperCut_D)

Data_1D = pd.read_csv(Data_1D_path, nrows=UpperCut_D)

Data_2D = pd.read_csv(Data_2D_path, nrows=UpperCut_D)

Data_3D = pd.read_csv(Data_3D_path, nrows=UpperCut_D)

Data_4D = pd.read_csv(Data_4D_path, nrows=UpperCut_D)

SamplesRemoved = 2

Data_0D = Data_0D.iloc[::2, :]
Data_1D = Data_1D.iloc[::2, :]
Data_2D = Data_2D.iloc[::2, :]
Data_3D = Data_3D.iloc[::2, :]
Data_4D = Data_4D.iloc[::2, :]


# Read the CSV file into DataFrame
Data_0E=pd.read_csv(Data_0E_path, nrows=UpperCut_E)

Data_1E=pd.read_csv(Data_1E_path, nrows=UpperCut_E)

Data_2E=pd.read_csv(Data_2E_path, nrows=UpperCut_E)

Data_3E=pd.read_csv(Data_3E_path, nrows=UpperCut_E)
Data_4E=pd.read_csv(Data_4E_path, nrows=UpperCut_E)

UpperCut = 3100000
#DataLength = Data_4D['Vibration_1'].shape[0]
Length = UpperCut-100000
DataFrame = pd.concat([Data_0D,Data_1D,Data_2D,
                       Data_3D, Data_4D 
                      ],ignore_index=True)

print(f"The data has been read now")

#DataFrame.info()

# Correcting data a bit:
# num_samples = DataFrame['Vibration_1'].shape[0]  # Replace this with the actual number of samples in your data
# sampling_frequency = 4096  # Replace this with the actual sampling frequency in Hz

# duration_seconds = 20
# samples_to_cut = int(duration_seconds * sampling_frequency)

# # Calculate time corresponding to each sample
# time = [sample / sampling_frequency for sample in range(num_samples)]
# time = np.array(time)

# #DataFrame.hist(bins=100, figsize=(20,15))

# # The data needs to get processed further to be plotable
# PlottingRange = np.linspace(0, 800000, 1001, dtype=int)
# plt.scatter(time[PlottingRange],DataFrame['Vibration_1'][PlottingRange])

# plt.scatter(DataFrame['Force'][PlottingRange],DataFrame['Vibration_1'][PlottingRange])
# plt.plot(DataFrame['Vibration_1'])

##########

# ## Add more signals
# DataFrame['Ramp_Vibration_3'] = DataFrame['Vibration_3'].diff()
# DataFrame['Ramp_Vibration_3'].iloc[0] = 0

# Set every 500,000th entry to 0
#DataFrame['Ramp_Vibration_3'].iloc[::500000] = 0

# Adding rolling mean
DataFrame['Meaned_Vib_3'] = DataFrame['Vibration_3'].rolling(window=round(Length/366), min_periods=1).mean()

# Adding rolling std
DataFrame['Std_Vib_3'] = DataFrame['Vibration_3'].rolling(window=round(Length/800), min_periods=1).std()
DataFrame['Std_Vib_3'].iloc[0] = 0

# Adding rolling max
DataFrame['max_Vib_3'] = DataFrame['Vibration_3'].rolling(window=round(Length/120000), min_periods=1).max()

DataFrame['Labels'] = (DataFrame.index // Length) % 5


#X_train = DataFrame[['Vibration_3','Std_Vib_3','max_Vib_3']].values
#y_train = DataFrame['Labels'].values



# Add min max scaler
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler

# features_to_normalize = ['Meaned_Vib_3', 'Measured_RPM', 'Std_Vib_3', 'max_Vib_3', 'Vibration_3',
#                          'V_in','Vibration_1','Vibration_2']

# scaler = MinMaxScaler()
# DataFrame[features_to_normalize] = scaler.fit_transform(DataFrame[features_to_normalize])


# corr_matrix =  DataFrame.corr()

# corr_matrix['Labels'].sort_values(ascending=False)

# Sequence the data
sampling_frequency = 4096  # Replace this with the actual sampling frequency in Hz
duration_seconds = 0.5
samples_to_cut = int(duration_seconds * sampling_frequency)


def features(data):
    n = int(np.floor(len(data['Vibration_3']) / samples_to_cut))
    data = data.iloc[:n * samples_to_cut]
    
    X = np.zeros((n, samples_to_cut, 6))  # Initialize an array for 5 features
    # X[:, :, 0] = data['V_in'].values.reshape((n, samples_to_cut))
    X[:, :, 0] = data['Meaned_Vib_3'].values.reshape((n, samples_to_cut))
    X[:, :, 1] = data['Measured_RPM'].values.reshape((n, samples_to_cut))
    X[:, :, 2] = data['Std_Vib_3'].values.reshape((n, samples_to_cut))
    X[:, :, 3] = data['max_Vib_3'].values.reshape((n, samples_to_cut))
    X[:,:,4] = data['Vibration_3'].values.reshape((n, samples_to_cut))
    X[:,:,5] = data['Vibration_2'].values.reshape((n, samples_to_cut))
    #X[:,:,6] = data['Vibration_1'].values.reshape((n, samples_to_cut))
    #X[:,:,7] = data['V_in'].values.reshape((n, samples_to_cut))
    y = data['Labels'].values.reshape((n, samples_to_cut))
    
    return X,y


X_train,y_train = features(DataFrame)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
y_train = y_train[:,1]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size = 0.2, random_state = 0)



#NrSeq = 3000

# SequenceLength = int(num_samples/NrSeq)



# X_train = np.zeros([NrSeq,SequenceLength,6])
# y_train = np.zeros(NrSeq)
# z = 0
# for i in range(NrSeq-1):
    
#     y_train[i] = DataFrame['Labels'][z]
    
#     for j in range(SequenceLength-1):
            
#         X_train[i,j,:] = DataFrame.iloc[z,0:6]
#         z = z+1
      
print(f"Training the model...")      
        
from sklearn.linear_model import SGDClassifier

from sklearn import svm

model = svm.SVC(gamma=0.001)

# Fit the model
model.fit(X_train, y_train)


print(f"Training complete")
#model = SGDClassifier()


# # Now X_train_reshaped has a shape of (1000, 2500) assuming each sample has shape (500, 5)

# # Instantiate the SGDClassifier
# model = SGDClassifier()
# svmModel = svm.SVC(max_iter=1000, verbose=True)

# model = SGDClassifier()

# tuning_parameters = {
#     'learning_rate': ('constant', 'optimal','adaptive'), 
#     'eta0': [0.01, 0.1, 1, 5, 10],
#     'penalty': ('l2', 'elasticnet', None),
#     'warm_start': (False,True)
# }


# CV = 3
# VERBOSE = 0

# # Run GridSearchCV for the model
# grid_tuned = GridSearchCV(model,
#                           tuning_parameters,
#                           cv=CV,
#                           scoring='f1_micro',
#                           verbose=VERBOSE,
#                           n_jobs=-1)

# num_iterations = 10
# f1_grid_results = []
# for i in range(num_iterations):
#     print(i)

#     # Setup search parameters
#     model = SGDClassifier()

#     tuning_parameters = {
#         'learning_rate': ('constant', 'optimal','adaptive'), 
#         'eta0': [0.01, 0.1, 1, 5, 10],
#         'penalty': ('l2', 'l1', 'elasticnet', None),
#         'warm_start': (False,True),
#         'average': [0.01,0.1,1,2,5]
#     }


#     CV = 5
#     VERBOSE = 0

#     # Run GridSearchCV for the model
#     grid_tuned = RandomizedSearchCV(model,
#                               tuning_parameters,
#                               n_iter=(i+1)*2,
#                               random_state=42, 
#                               cv=CV,
#                               scoring='f1_micro',
#                               verbose=VERBOSE,
#                               n_jobs=-1)

    
#     grid_tuned.fit(X_train, y_train)
    
    
#     f1_grid_results.append(grid_tuned.best_score_) 


# Fit the model
#model.fit(X_train, y_train)
#svmModel.fit(X_train, y_train)
# grid_tuned.fit(X_train, y_train)

################# Setting up the test data #######################
     
#DataLength = Data_4E['Vibration_1'].shape[0]
UpperCut = 700000
Length = UpperCut-100000
DataFrame_E = pd.concat([Data_0E[100000:UpperCut],Data_1E[100000:UpperCut], 
                   Data_2E[100000:UpperCut],
                   Data_3E[100000:UpperCut],
                   Data_4E[100000:UpperCut] 
                  ],ignore_index=True)    


DataFrame_E = DataFrame_E.drop(['Force'],axis=1)


# ## Add more signals
# DataFrame['Ramp_Vibration_3'] = DataFrame['Vibration_3'].diff()
# DataFrame['Ramp_Vibration_3'].iloc[0] = 0

# Set every 500,000th entry to 0
#DataFrame_E['Ramp_Vibration_3'].iloc[::500000] = 0

# Adding rolling mean
DataFrame_E['Meaned_Vib_3'] = DataFrame_E['Vibration_3'].rolling(round(Length/366), min_periods=1).mean()

# Adding rolling std
DataFrame_E['Std_Vib_3'] = DataFrame_E['Vibration_3'].rolling(window=round(Length/800), min_periods=1).std()
DataFrame_E['Std_Vib_3'].iloc[0] = 0


# Adding rolling max
DataFrame_E['max_Vib_3'] = DataFrame_E['Vibration_3'].rolling(window=round(Length/120000), min_periods=1).max()


DataFrame_E['Labels'] = (DataFrame_E.index // Length) % 5

#DataFrame.hist(bins=75, figsize=(20,15))  

# corr_matrix_E =  DataFrame_E.corr()

# corr_matrix_E['Labels'].sort_values(ascending=False)



# scaler = MinMaxScaler()
# DataFrame_E[features_to_normalize] = scaler.fit_transform(DataFrame_E[features_to_normalize])


X_test,y_test = features(DataFrame_E)
X_test = np.reshape(X_test, (X_test.shape[0], -1))
y_test = y_test[:,1]


y_pred = model.predict(X_test)


plt.plot(y_pred)
plt.plot(y_test)
plt.title('SVC prediction matched with true values')
plt.legend(['SVC predictions','True values'])


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, precision_score, recall_score

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = model.score(X_test, y_test)
#accuracy_val= model.score(X_test_val, y_test_val)
f1score = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")
print(f"Accuracy: {accuracy}")
#print(f"Accuracy_val: {accuracy_val}")
print(f"F1-score: {f1score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


####################


# # Exploring data further

# from mpl_toolkits.mplot3d import Axes3D

# # Assuming df is your DataFrame
# fig = plt.figure(figsize=(20, 15))
# ax = fig.add_subplot(111, projection='3d')

# # Extract the three columns you want to use for the 3D plot
# x = DataFrame['Std_Vib_3']
# y = DataFrame['max_Vib_3']
# z = DataFrame['Labels']

# # Create a 3D scatter plot
# ax.scatter(x, y, z, c='blue', marker='o',alpha=0.3)

# # Set labels for each axis
# ax.set_xlabel('Std_Vib_3')
# ax.set_ylabel('max_Vib_3')
# ax.set_zlabel('Labels')

# # Set the plot title
# ax.set_title('3D Scatter Plot Train')

# fig = plt.figure(figsize=(20, 15))
# ax = fig.add_subplot(111, projection='3d')

# # Extract the three columns you want to use for the 3D plot
# x = DataFrame_E['Std_Vib_3']
# y = DataFrame_E['max_Vib_3']
# z = DataFrame_E['Labels']

# # Create a 3D scatter plot
# ax.scatter(x, y, z, c='blue', marker='o',alpha=0.3)

# # Set labels for each axis
# ax.set_xlabel('Std_Vib_3')
# ax.set_ylabel('max_Vib_3')
# ax.set_zlabel('Labels')

# # Set the plot title
# ax.set_title('3D Scatter Plot Test')


 
# DataFrame_E.hist(bins=75, figsize=(20,15))   



# num_samples_E = DataFrame_E['Vibration_1'].shape[0]


# NrSeq_E = 6000

# SequenceLength_E = int(num_samples_E/NrSeq_E)


# X_test = DataFrame_E[['Vibration_3','Std_Vib_3','max_Vib_3']].values
# y_test = DataFrame_E['Labels'].values

# X_test = np.zeros([NrSeq_E,SequenceLength_E,6])
# y_test = np.zeros(NrSeq_E)
# z = 0
# for i in range(NrSeq_E-1):
    
#     y_test[i] = DataFrame_E['Labels'][z]
    
#     for j in range(SequenceLength_E-1):
            
#         X_test[i,j,:] = DataFrame_E.iloc[z,0:6]
#         z = z+1

# X_test_reshaped = X_test.reshape((X_test.shape[0], -1))

# y_pred = model.predict(X_test)
# y_pred = grid_tuned.predict(X_test)

# plt.plot(y_pred)
# plt.plot(y_test)
# plt.title('SGD prediction matched with true values')
# plt.legend(['SGD predictions','True values'])