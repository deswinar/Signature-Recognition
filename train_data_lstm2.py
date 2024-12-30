import socket, pickle
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
    configur = ConfigParser()
    print(configur.read('config.ini'))

    min_loss = configur.getfloat('training', 'min_loss')
    server_ip = configur.get('training', 'server_ip')

    # Send dataset to be trained on server
    # Server IP and Port to connect
    HOST = server_ip
    PORT = 50007
    # Create a socket connection.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((HOST, PORT))
        BASE_PATH = ""
        
        base_path = f"dataset/{username}"
        os.makedirs(base_path, exist_ok=True)
        
        filenames = [f"{base_path}/axis_x_train",
                     f"{base_path}/axis_y_train"]
        with open(f"{filenames[0]}.txt", "r") as dataset_x, \
             open(f"{filenames[1]}.txt", "r") as dataset_y:
            dataset = list()
            
            data_with_labels_x = np.genfromtxt(dataset_x)
            data_with_labels_y = np.genfromtxt(dataset_y)
            dataset.append(data_with_labels_x)
            dataset.append(data_with_labels_y)
            labels = data_with_labels_x[:,0]

            dataset = np.dstack(dataset)
            print(f"Dataset : {dataset[:,1:,:].shape}")
            print(f"Labels : {dataset[:,0,0].shape}")

            flatten_dataset = dataset.flatten()
            list_dataset = list(flatten_dataset)
            str_dataset = str(list_dataset)

        dataset_size = f"{str(len(str_dataset))}_{dataset.shape}"
        print(dataset_size)
        s.send(dataset_size.encode())
        s.send(str_dataset.encode())

        # Receive pytorch model from server
        data = b""
        while True:
            packet = s.recv(999999)
            if not packet: break
            data += packet

        model = pickle.loads(data)
        print ('Pytorch model received from server')
        print (model)

        s.close()

        # Specify a path for saving model
        MODELS_PATH = 'models'
        if not os.path.exists(MODELS_PATH):
            os.mkdir(MODELS_PATH)
        model_filename = f'/lstm_model_{username}.pt'

        # Save nn model
        torch.save(model, MODELS_PATH+model_filename)
    except socket.error as err:
        print(f"Server error : {err}")
        print("Start training using this PC instead")
        #model = train_dataset(self.username)
        # Save nn model
        #torch.save(model, MODELS_PATH+model_filename)
    return "ok"

# Run only if in this namespace
if __name__ == '__main__':
    test = train_dataset('test')
