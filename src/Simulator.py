import pandas as pd
import numpy as np
import sys 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import itertools
from lightgbm import LGBMClassifier

from joblib import Parallel, delayed, load

import pickle

from abc import abstractmethod, ABC
import json
from .DataManager import BetterTokenizer

import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time


# with open('test_tuple.pkl', 'rb') as file:
#     loaded_tuple = pickle.load(file)




# loaded_tuple = pd.read_pickle('src/data/test_tuple.pkl')
X_test = np.load('src/data/test_features.npz')['a']
y_test = np.load('src/data/test_labels.npz')['a']
token_items = json.load(open('src/data/token1.json'))



scaled_X_test = np.load('src/data/scaled_test_features.npz')['a']
scaled_y_test = np.load('src/data/scaled_test_labels.npz')['a']




class ScaledSampler(ABC):
    def __init__(self) -> None:
        self.testing_data = scaled_X_test
        self.testing_data = self.testing_data[:,::-1].copy()
        self.testing_truths = scaled_y_test

        self.sample_length = len(scaled_X_test)

        self.features = self.testing_data.reshape(-1, 47, 1)
    
    def get_sample(self, input_int):
        sample = self.features[input_int]
        tensor_sample = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)
        return tensor_sample

        




class TestingSampler(ABC):
    def __init__(self):

        self.testing_data = X_test
        
        self.testing_truths = y_test
        self.sample_length = len(X_test)
        self.scaler = MinMaxScaler(feature_range=(0,1))

        self.unscaled_lstm_features = self.testing_data.copy().reshape(-1, 47, 1)
        # features = self.scaler.fit_transform(self.testing_data.copy())
        features = self.testing_data.reshape(-1, 47, 1)
        self.lstm_y_val = features.reshape(-1, 47, 1)


        self.ls_sampler = ScaledSampler()


    def update_seed(self):
        np.random.seed(np.random.randint(2,200000))

    
    #sample from the length of testing data
    def get_sample(self):
        sample_int = np.random.choice(self.sample_length, replace=False)
        self.sample_int = sample_int
        sample = (self.testing_data[sample_int], self.testing_truths[sample_int])
        self.sample = sample
        return sample[0] 
    
    # def get_lstm_sample(self):
    #     y_val =  self.lstm_y_val[self.sample_int]
    #     y_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(0)
    #     return y_tensor

    def get_lstm_sample(self):
        sample = self.ls_sampler.get_sample(input_int=self.sample_int)
        return sample
        

    

class TargetManager:

    def __init__(self) -> None:
        self.target_keys = {
            'benign':0,
            'phishing':1,
            'defacement':2,
            'malware':3
        }

        self.int_target = {value:key for key, value in self.target_keys.items()}



# input_dim = 1
# # hidden_dim = 128

# hidden_dim = 42
# output_dim = 4  # Assuming binary classification
# num_layers = 6


    

# model = LSTM(input_size=input_dim, hidden_size=hidden_dim, layers=num_layers).to(device)



    
class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(StackedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_size, output_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    


# input_dim = 1
# hidden_dim = 46
# output_dim = 4 
# num_layers = 2



# input_dim = 1
# hidden_dim = 256
# output_dim = 4  
# num_layers = 3
# hidden_dim = 512

# learing_rate = 0.001







class ScoreNode:
    def __init__(self):
        self.reset()

    def reset(self):
        self.right = 0
        self.wrong = 0

class LSTMAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMAttentionClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, num_classes)  # Adapted for classification

        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc(context)
        return out



input_dim = 1
hidden_dim = 96
output_dim = 4  # Assuming binary classification
num_layers = 2


learing_rate = 0.01
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # We multiply by 2 for bidirectional

        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Assuming x is already in the correct shape: [batch_size, seq_len, input_dim]
        lstm_out, (hidden, cell) = self.lstm(x)
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc2(hidden)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    



class Simulator(TargetManager):

    def __init__(self) -> None:
        super().__init__()
        self.sampler = TestingSampler()
        self.model = load('src/data/gb_model.joblib')

        # self.lstm = LSTM().to(device)
        # self.lstm = LSTMAttentionClassifier(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, num_classes=output_dim).to(device)
        #self.lstm = StackedLSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, output_size=output_dim, dropout_rate=0.01).to(device)
        self.lstm = BiLSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
        self.lstm.load_state_dict(torch.load('src/data/model2.pt'))
        #self.lstm.load_state_dict(torch.load('code/data/model.pt'))
        self.lstm.eval()

        self.tokenizer = BetterTokenizer(starting_tokenizer=token_items)
        self.score_node = ScoreNode()
        self.gb_node = ScoreNode()

    def lsmt_predict(self, X_test):
        outputs = self.lstm(X_test)
        preds = outputs.argmax().item()
        return preds
    

    
    def run(self):
        self.score_node.reset()
        for i in range(50):
            sample = self.sampler.get_sample()
            url = self.tokenizer.decode_padded(sample)

            prediction = self.model.predict(sample.reshape(1, -1))[0]
            prediction_string = self.int_target.get(prediction)
            actual = self.int_target.get(self.sampler.sample[1])
            actual_int = self.sampler.sample[1]

            lstm_sample = self.sampler.get_lstm_sample()
            lstm_prediction = self.lsmt_predict(lstm_sample)

            if (actual_int == 1) & (lstm_prediction == 1):
                self.score_node.right += 1
            elif (actual_int == 1) & (lstm_prediction == 0):
                self.score_node.wrong += 1
            
            if (prediction == 1) & (actual_int == 1):
                self.gb_node.right += 1
            elif (prediction == 1) & (actual == 0):
                self.gb_node.wrong += 1
            


            




            
            print(f'Model is predicting the URL {url} to be \n {prediction_string}')
            print(f'The LSTM predicted {self.int_target.get(lstm_prediction)}')
            print(f'The Actual URL is {actual}')
            print('-' * 75)
            time.sleep(0.5)
        print(f'LSTM correctly predicted {self.gb_node.right} out of {self.gb_node.right + self.gb_node.wrong}')
            # prediction = self.model.predict(self.sampler.sample)
            # print(f'The Cyber Model predicts the url of {self.sampler.sample} to be {prediction}')



        