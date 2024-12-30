import numpy as np
import torch
import torch.nn as nn

NUM_INPUT_FEATURES = 0

CONV_FILTERS = 25
CONV_FILTERS2 = 10
CONV_FILTER_SIZE = 9

device = 'cpu'

class Net1_CNN(nn.Module):
    def __init__(self):
        super(Net1_CNN, self).__init__()
        num_units_fc = 100
        self.num_inputs_fc = int(CONV_FILTERS*(NUM_INPUT_FEATURES-CONV_FILTER_SIZE+1)/2)

        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = CONV_FILTERS, 
                               kernel_size=CONV_FILTER_SIZE, padding = 0, stride = 1)
        self.max_pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(self.num_inputs_fc, num_units_fc)
        self.relu = nn.ReLU()
        self.out = nn.Linear(num_units_fc, NUM_CLASSES) 

    def forward(self, x):
        x = x.view(-1, 1, NUM_INPUT_FEATURES)
        x = self.conv1(x)
        x = self.max_pool(x)
        x = x.view(-1, self.num_inputs_fc)
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)
        return x


class Net2_CNN(nn.Module):
    def __init__(self):
        super(Net2_CNN, self).__init__()
        num_units_fc = 100
        self.num_inputs_fc = int(CONV_FILTERS2*((NUM_INPUT_FEATURES-CONV_FILTER_SIZE+1)/2-CONV_FILTER_SIZE+1)/2)

        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = CONV_FILTERS, 
                               kernel_size=CONV_FILTER_SIZE, padding = 0, stride = 1)
        self.max_pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(in_channels = CONV_FILTERS, out_channels = CONV_FILTERS2, 
                               kernel_size=CONV_FILTER_SIZE, padding = 0, stride = 1)
        self.max_pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(self.num_inputs_fc, num_units_fc)
        self.relu1 = nn.ReLU()
        self.out = nn.Linear(num_units_fc, NUM_CLASSES) 

    def forward(self, x):
        x = x.view(-1, 1, NUM_INPUT_FEATURES)
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = x.view(-1, self.num_inputs_fc)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.out(x)
        return x


# Please note that the dynamic convolutional layer is initialized using the 
# parameters learned during the "pre-train" phase (in which a "usual" 
# convolutional network is trained). Once the pre-train phased is completed, 
# the parameters of the dynamic convoltuional layer are fixed, therefore,
# the activations of the dynamic convolutional layer will be pre-calculated 
# outside the network for efficient implementation.

class Net1_DCNN(nn.Module):
    def __init__(self):
        super(Net1_DCNN, self).__init__()
        num_units_fc = 100

        self.max_pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(int(CONV_FILTERS*(NUM_INPUT_FEATURES-CONV_FILTER_SIZE+1)/2), num_units_fc)
        self.relu = nn.ReLU()
        self.out = nn.Linear(num_units_fc, NUM_CLASSES) 

    def forward(self, x):
        x = x.view(-1, CONV_FILTERS, NUM_INPUT_FEATURES-CONV_FILTER_SIZE+1)
        x = self.max_pool(x)
        x = x.view(-1,int(CONV_FILTERS*(NUM_INPUT_FEATURES-CONV_FILTER_SIZE+1)/2))
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)
        return x


class Net2_DCNN(nn.Module):
    def __init__(self):
        super(Net2_DCNN, self).__init__()
        num_units_fc = 100
        self.num_inputs_fc = int(CONV_FILTERS2*((NUM_INPUT_FEATURES-CONV_FILTER_SIZE+1)/2-CONV_FILTER_SIZE+1)/2)

        self.max_pool = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(in_channels = CONV_FILTERS, out_channels = CONV_FILTERS2, 
                               kernel_size=CONV_FILTER_SIZE, padding = 0, stride = 1)
        self.max_pool2 = nn.MaxPool1d(2)
        self.fc = nn.Linear(self.num_inputs_fc, num_units_fc)
        self.relu = nn.ReLU()
        self.out = nn.Linear(num_units_fc, NUM_CLASSES) 
 

    def forward(self, x):
        #x = x.view(-1, CONV_FILTERS, NUM_INPUT_FEATURES-CONV_FILTER_SIZE+1)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = x.view(-1, self.num_inputs_fc)
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)
        return x

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, drop_prob=0.5):
        super(LSTMNet, self).__init__()
        # output_size = NUM_CLASSES
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden states and cell states for LSTM
        # h0.shape: (num_layers, batch_size, hidden_size)
        # c0.shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # x.shape : (batch_size, seq_length, input_size)
        #batch_size = x.size(0)

        # lstm_out.shape: (batch_size, seq_length, hidden_size)
        lstm_out, hidden = self.lstm(x, (h0,c0))
        out = self.dropout(lstm_out)

        # Decode the hidden state of the last time step
        # out.shape: (batch_size, hidden_size)
        out = lstm_out[:, -1, :]

        # out.shape = (batch_size, output_size)
        out = self.fc(out)
        
        return out

class ConvLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, drop_prob=0.1):
        super(ConvLSTMNet, self).__init__()
        # output_size = NUM_CLASSES
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.conv = nn.Conv1d(2, output_size, 12, stride=1)
        self.max_pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        #self.fc = nn.Softmax()
        #self.fc = nn.Sigmoid()

        #self.fc2 = nn.Sigmoid()

    def forward(self, x):
        # Set initial hidden states and cell states for LSTM
        # h0.shape: (num_layers, batch_size, hidden_size)
        # c0.shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        #print(f'x.shape = {x.shape}')

        # x.shape : (batch_size, seq_length, input_size)

        x = torch.swapaxes(x, 1, 2)
        x = self.conv(x)
        x = self.max_pool(x)
        x = torch.swapaxes(x, 1, 2)

        # lstm_out.shape: (batch_size, seq_length, hidden_size)
        lstm_out, hidden = self.lstm(x, (h0,c0))
        #print(f'lstm_out.shape = {lstm_out.shape}')

        out = lstm_out
        out = self.dropout(lstm_out)
        #print(f'out.shape = {out.shape}')

        # Decode the hidden state of the last time step
        # out.shape: (batch_size, hidden_size)
        out = out[:, -1, :]
        #print(f'out.shape = {out.shape}')

        # out.shape = (batch_size, output_size)
        out = self.fc(out)
        #out = torch.sigmoid(self.fc(out))
        #print(f'out.shape = {out.shape}')
        #out = self.fc2(out)
        
        return out

class ConvNet(nn.Module):
    def __init__(self, input_size, seq_len, output_size):
        super(ConvNet, self).__init__()
        # output_size = NUM_CLASSES
        self.input_size = input_size
        self.seq_len = seq_len
        self.output_size = output_size
        #self.in_fc = int(2*((self.seq_len-2+1)/2-12+1)/2/2)
        self.in_fc = 41

        self.conv = nn.Conv1d(self.input_size, 2, 12, stride=1)
        self.max_pool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(2, 2, 12, stride=1)
        self.max_pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(self.in_fc, output_size)
        #self.fc = nn.Softmax()
        #self.fc = nn.Sigmoid()

        #self.fc2 = nn.Sigmoid()

    def forward(self, x):
        #print(f'x.shape = {x.shape}')

        # x.shape : (batch_size, seq_length, input_size)

        x = torch.swapaxes(x, 1, 2)
        x = self.conv(x)
        x = self.max_pool(x)
        #print(f"conv : {x.shape}")
        x = self.conv1(x)
        x = self.max_pool(x)
        #print(f"conv1 : {x.shape}")
        out = self.fc(x)
        #print(f"fc : {out.shape}")
        out = out[:, -1, :]
        
        return out
