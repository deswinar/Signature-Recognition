import socket, pickle

import re
import os

from models import *
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def train_dataset(conn, data_with_labels, min_loss):
    min_loss = min_loss
    epochs = 1000
    losses = []
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    data = data_with_labels[:,1:,:]
    labels = data_with_labels[:,0,0]

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

    batch_size = 300

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    input_size = 2
    output_size = NUM_CLASSES
    hidden_size = 128
    num_layers = 3

    lr=0.0001

    #epochs = 240
    print_every = 10

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
                #conn.send(f"INFO_Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}".encode('utf-8'))
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
    MODELS_PATH = 'models'
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)
    model_filename = f'{MODELS_PATH}/temp_model.pt'

    # Save nn model
    torch.save(model, model_filename)

    return model

hostname = socket.gethostname()
server_ip = socket.gethostbyname(hostname)

HOST = server_ip
PORT = 50007
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print(f"{HOST}:{PORT}. Listening...")
while True:
    conn, addr = s.accept()
    print ('Connected by', addr)

    # Receive dataset from Raspberry
    packet_size = conn.recv(1024).decode()
    packet_size = str(packet_size).split('_')
    print(f"packet_size : {packet_size[0]}")
    print(f"dataset.shape : {packet_size[1]}")
    print(f"min_loss : {packet_size[2]}")
    dataset_shape = eval(packet_size[1])
    min_loss = float(packet_size[2])
    data = ""
    while True:
        packet = conn.recv(99999999).decode()
        
        data += packet
        if not packet or len(data) >= int(packet_size[0]):
            break
        
    print(len(data))
    
    data = re.sub(r"[\[\]]", "", data)
    data = data.split(',')
    #np.frombuffer(bytes_np_dec, dtype=np.float64)
    
    data_with_labels = np.array(data, dtype=float)
    #data_with_labels = pickle.loads(data)
    print ('Dataset received from Raspberry')
    print (data_with_labels.shape)
    data_with_labels = data_with_labels.reshape(dataset_shape)

    model = train_dataset(conn, data_with_labels, min_loss)

    # Send pytorch model to Raspberry
    #model = torch.load("model.pt")
    # Pickle the object and send it to Raspberry
    model_string = pickle.dumps(model)
    conn.send(model_string)
    print ('Pytorch model sent to Raspberry')
    print(model)

    print("Listening...")
    conn.close()

