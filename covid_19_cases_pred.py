# %%
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, datetime, pickle

# %%
# Step 1: Data Loading
# train dataset
TRAIN_CSV_PATH = os.path.join(os.getcwd(), 'dataset', 'cases_malaysia_train.csv')
train_df = pd.read_csv(TRAIN_CSV_PATH)

# test dataset
TEST_CSV_PATH = os.path.join(os.getcwd(), 'dataset', 'cases_malaysia_test.csv')
test_df = pd.read_csv(TEST_CSV_PATH)

# %%
# Step 2: Data Inspection and Visualization
# train dataset
train_df.info()     # train_df['cases_new'] is object

# to convert train_df['cases_new'] from object into numeric
train_df['cases_new'] = pd.to_numeric(train_df['cases_new'], errors='coerce')

# %%
# to check if the previous operation was a success
train_df.info()

print(train_df['cases_new'].isna().sum())   # 12 null/NaN

# To show the trend and anomalies lie in the data
plt.figure()
plt.plot(train_df['cases_new'])
plt.show()

# %%
# To check any other anomalies in data
print(train_df.head(5))
# %%
print(train_df.tail(5))

# %%
# test dataset
test_df.info()
print(test_df['cases_new'].isna().sum())    # 1 null/NaN

# To show the trend and anomalies lie in the data
plt.figure()
plt.plot(test_df['cases_new'])
plt.show()

# %%
# To check any other anomalies in data
print(train_df.head(5))
# %%
print(train_df.tail(5))

# %%
# Step 3: Data Cleaning
# train dataset
# To replace the NaNs with Interpolation
train_df['cases_new'] = train_df['cases_new'].interpolate(method='polynomial', order=2)

# to check if the previous operation was a success
plt.figure()
plt.plot(train_df['cases_new'])
plt.show()

# %%
# test dataset
# To replace the NaNs with Interpolation
test_df['cases_new'] = test_df['cases_new'].interpolate(method='polynomial', order=2)

# to check if the previous operation was a success
plt.figure()
plt.plot(test_df['cases_new'])
plt.show()

# %%
# Step 5: Data Preprocessing
train_df['cases_new'] = train_df['cases_new'].astype(int)      
data = train_df['cases_new'].values
data = data[::, None]

# %%
# Normalization
mms = MinMaxScaler()
mms.fit(data)
data = mms.transform(data)

# %%
win_size = 30
X_train = []
y_train = []

for i in range(win_size, len(data)) :
    X_train.append(data[i-win_size:i])
    y_train.append(data[i])
    
# to convert into numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=123)

# %%
# Model Development using Sequential API
model = Sequential()

model.add(LSTM(64, input_shape=X_train.shape[1:]))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()

plot_model(model, show_shapes=True)

model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape'])

# callbacks
LOG_DIR = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = TensorBoard(log_dir=LOG_DIR)
es = EarlyStopping(monitor= 'val_loss', patience=10)

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[tb, es])

# %%
test_df['cases_new'] = test_df['cases_new'].astype(int)   
concat = pd.concat((train_df['cases_new'], test_df['cases_new']))
concat = concat[len(concat) - win_size - len(test_df):]

concat = mms.transform(concat[::, None])

X_testtest = []
y_testtest = []

for i in range(win_size, len(concat)) :
    X_testtest.append(concat[i-win_size:i])
    y_testtest.append(concat[i])

# to convert into numpy array
X_testtest = np.array(X_testtest)
y_testtest = np.array(y_testtest)

# %%
# to predict the Covid19 cases based on the testing dataset
predicted_cases = model.predict(X_testtest)

# %%
# To visualize the predicted cases and actual cases
plt.figure()
plt.plot(predicted_cases, color='r')
plt.plot(y_testtest, color='b')
plt.legend(['Predicted', 'Actual'])
plt.xlabel('Day')
plt.ylabel('Number of Cases')
plt.show()

# metrics to evaluate the performance
mape = mean_absolute_percentage_error(y_testtest, predicted_cases)
mae = mean_absolute_error(y_testtest, predicted_cases)

print(f'The Mean Absolute Percentage Error (MAPE) for this Covid19 case prediction is {mape} and the Mean Absolute Error is {mae}' )

# %%
# Plot Graph
train_loss = history.history['loss']
test_loss = history.history['val_loss']
train_mape = history.history['mape']
test_mape = history.history['val_mape']
epoch_no = history.epoch

plt.plot(epoch_no, train_loss, label='Training Loss')
plt.plot(epoch_no, test_loss, label='Validation Loss')
plt.legend()
plt.title('Loss Graph')
plt.show()

plt.plot(epoch_no, train_mape, label='Training MAPE')
plt.plot(epoch_no, test_mape, label='Validation MAPE')
plt.legend()
plt.title('Evaluation Graph')
plt.show()

# %%
# Save MMS 
with open('mms.pkl', 'wb') as f :
    pickle.dump(mms, f)
    
# Save model
model.save('model.h5')


