import alphien
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
# Creates a data loader for UBS pricing data

d = alphien.data.DataLoader()

# Need to normalize the data for the neural net to interpret them properly
def engineerData(df):
    x = df.values #returns a numpy array
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

def splitXY(df):
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]
    return X,Y

# Keras sequence, like a generator but with the guarantee that the network will only train once on each sample per epoch
class ComplexPricingSequence(Sequence):

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.size = data.size()

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def __getitem__(self, index):
        batch = engineerData(next(self.data.batch(fromRow=index,toRow=index+self.batch_size)))
        batch_x, batch_y = splitXY(batch)

        return batch_x.to_numpy(), batch_y.to_numpy()

def training_loop():
    model = Sequential()
    model.add(Dense(164, input_dim=164, kernel_initializer='normal',
                   activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(109, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    dataSequence = ComplexPricingSequence(d,10000)
    model.fit(dataSequence,epochs=5,verbose=1)
    model.save('deepLearningModel.txt')

training_loop()

model = keras.models.load_model('deepLearningModel.txt')

dataTest = next(d.batch(fromRow=1, toRow=200000))
df = dataTest

def predictFunc(newData, dataTransformFunc, model):
    X,Y = splitXY(engineerData(newData))
    return model.predict(X)

predictFunc(dataTest, engineerData, model)

def evalFunc(newData, dataTransformFunc, model):
    X,Y = splitXY(engineerData(newData))
    return model.evaluate(X,Y)

evalFunc(dataTest, engineerData, model)
