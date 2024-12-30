import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import os

from models import ConvLSTMNet, ConvNet

from configparser import ConfigParser

##def train_dataset(window, username):
def train_dataset(username):
    # Base path to main folder
    # Raspberry
    BASE_PATH = r"/home/sulthon/Downloads/HandsignRecognition/"
    # Others
    BASE_PATH = ""
        
    configur = ConfigParser()
    print(configur.read(f'{BASE_PATH}config.ini'))

    min_loss = configur.getfloat('training', 'min_loss')

    epochs = 2000
    losses = []
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    DATASET_PATH = fr"{BASE_PATH}dataset/{username}/"
    file_name_prefix = ["axis_x", "axis_y"]

    # Load train and test data
    data = list()
    labels = list()
    for fnp in file_name_prefix:
      data_with_labels = np.genfromtxt(DATASET_PATH + fnp + '_train.txt')
      data.append(data_with_labels[:,1:])
      labels = data_with_labels[:,0]
      #labels.append(data_with_labels[:,0])

    data = np.dstack(data)
    #labels = np.vstack(labels)

    print(f"data.shape : {data.shape}")
    print(f"labels.shape : {labels.shape}")

    # We make sure that labels are numbered as 0, 1, 2, ... 
    # and set the number of classes

    min_label = min(labels)
    max_label = max(labels)
    if min_label == 0:
      NUM_CLASSES = int(max_label+1)
    elif min_label == 1:
      labels = labels - min_label
      NUM_CLASSES = int(max_label)
    elif min_label == -1:
      if np.sum(labels == -1)+np.sum(labels==1) == len(labels):
        NUM_CLASSES = 2
        labels[labels==-1]=0
      else:
        raise Exception("Unexpected labels")
    else:
      raise Exception("Unexpected labels")

    # Set number of input features
    NUM_INPUT_FEATURES = len(data[0])

    print(f"NUM_CLASES : {NUM_CLASSES}")
    print(f"NUM_INPUT_FEATURES : {NUM_INPUT_FEATURES}")

    train_data = TensorDataset(torch.Tensor(data), torch.LongTensor(labels))

    batch_size = 100

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    print(len(train_loader))
    for inputs, labels in train_loader:
        print(labels.shape)

    input_size = 2
    output_size = NUM_CLASSES
    seq_len = NUM_INPUT_FEATURES
    hidden_size = 100
    num_layers = 5

    lr=0.0001

    #epochs = 240
    print_every = 10

    #model = ConvNet(input_size, seq_len, output_size).to(device)
    #model = LSTMNet(input_size, hidden_size, num_layers, output_size).to(device)
    model = ConvLSTMNet(input_size, hidden_size, num_layers, output_size).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #losses = []
    n_total_steps = len(train_loader)
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
            #labels = labels.view(len(labels), 1).long()
            #print(labels.shape)

            outputs = model(inputs)
            #print(outputs)
            loss = criterion(outputs, labels)
            #loss = criterion(outputs, labels.reshape(-1,1).float())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % print_every == 0:
                #print(loss.detach().numpy().item())
                #losses.append(loss.detach().numpy())
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

            if loss.item() < min_loss:
                break
        else:
            continue
        break
    print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

##    for name, param in model.named_parameters():
##        print(f"{name} : {param}")

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            #print(outputs)

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on test inputs: {acc} %')


    # Specify a path for saving model
    MODELS_PATH = fr'{BASE_PATH}models'
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)
    model_filename = f'/lstm_model_{username}.pt'
    #model_filename = f'/conv_model_{username}.pt'

    # Save nn model
    torch.save(model, MODELS_PATH+model_filename)

    return model

# Run only if in this namespace
if __name__ == '__main__':
    test = train_dataset('test1')
