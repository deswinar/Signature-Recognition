import socket, pickle

import cv2
import numpy as np
import pandas as pd
import torch

import time
from datetime import timedelta
import os
import os.path
from pathlib import Path
import sys
import ctypes
from screeninfo import get_monitors

from configparser import ConfigParser

from train_data_lstm import train_dataset

np.set_printoptions(suppress=True, precision=6)

class CreateDataset:
    def __init__(self, username):
        self.username = username

        # Base path to main folder
        # Raspberry
        self.BASE_PATH = r"/home/sulthon/Downloads/HandsignRecognition/"
        # Others
        self.BASE_PATH = ""

        self.configur = ConfigParser()
        print(self.configur.read(f'{self.BASE_PATH}config.ini'))

        self.i_data = self.configur.getint('dataset', 'i_data')
        self.i_range = self.configur.getint('dataset', 'i_range')

##        self.i_data = 5 # Range of how many times dataset will be collected
##        self.i_range = 500 # Range of how many times axis will be recorded

        self.show_hint = True
        self.start_drawing = False # true if left mouse is pressed until reach i_range
        self.drawing = False # true if left mouse is pressed down
        self.pt1_x , self.pt1_y = None , None
        self.start_time = 0

        self.array_x, self.array_y = np.array([]), np.array([])
        self.interp_x, self.interp_y = np.array([]), np.array([])
        self.img = None
    
    def run(self):
        for myloop in range(self.i_data):
            self.show_hint = True
            self.start_drawing = False
            self.array_x, self.array_y = np.array([]), np.array([])
            
            # Fullscreen
            screen = get_monitors()[0]
            screen_width, screen_height = screen.width, screen.height
            self.img = np.zeros((screen_height,screen_width,3), np.uint8)
            cv2.putText(self.img, f"Draw your handsign", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Fullscreen
            cv2.namedWindow('img', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            cv2.setMouseCallback('img',self.line_drawing)
            # Getting the height and width of the image
            win_width = self.img.shape[1]
            win_height = self.img.shape[0]
            print(f"scren size : {win_width}, {win_height}")

            half_height = int(win_height/2)
            half_width = int(win_width/2)
            cv2.line(self.img, (0, int(half_height)), (win_width, half_height), (0, 0, 255), 1)
            cv2.line(self.img, (half_width, 0), (half_width, win_height), (0, 0, 255), 1)

            measure1 = time.time()
            measure2 = time.time()
            count = 0
            while(1):
                if self.start_drawing == True:
                    if self.show_hint == True:
                        self.img = np.zeros((screen_height,screen_width,3), np.uint8)
                        cv2.line(self.img, (0, int(half_height)), (win_width, half_height), (0, 0, 255), 1)
                        cv2.line(self.img, (half_width, 0), (half_width, win_height), (0, 0, 255), 1)
                        self.show_hint = False
                    if measure2 - measure1 >= 0.01:
                        count += 1
                        self.array_x = np.append(self.array_x, self.pt1_x)
                        self.array_y = np.append(self.array_y, self.pt1_y)

                        measure1 = measure2
                        measure2 = time.time()
                    else:
                        measure2 = time.time()

                    if count >= self.i_range:
                        self.drawing = False
                        self.start_drawing=False
                        end_time = time.time()
                        print(f"Elapsed Time : {timedelta(seconds=end_time - self.start_time)}")
                        # Scale axis to (-2, 2) coordinates
                        self.interp_x = np.interp(self.array_x,[np.amin(self.array_x),np.amax(self.array_x)],[-2,2])
                        self.interp_y = np.interp(self.array_y,[np.amin(self.array_y),np.amax(self.array_y)],[2,-2])

                        # Call Write Data to Csv Function
                        self.writeCsv()
                        
                        # Add Noise to dataset
                        for i in range(3):
                            self.interp_x = np.interp(self.array_x-(5*i),[np.amin(self.array_x),np.amax(self.array_x)],[-2,2])
                            self.interp_y = np.interp(self.array_y-(5*i),[np.amin(self.array_y),np.amax(self.array_y)],[2,-2])
                            self.writeCsv()
                        break
                if myloop == self.i_data - 1 and count >= self.i_range - 2:
                    cv2.putText(self.img, f"Collecting data...Please wait 1-2 minutes", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow('img',self.img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        cv2.setMouseCallback('img', lambda *args : None)
        non_handsign_filenames = [fr"{self.BASE_PATH}dataset/non_handsign_x.txt",
                                  fr"{self.BASE_PATH}dataset/non_handsign_y.txt"]

        base_path = f"{self.BASE_PATH}dataset/{self.username}"
        os.makedirs(base_path, exist_ok=True)
        
        filenames = [fr"{base_path}/axis_x_train",
                     fr"{base_path}/axis_y_train"]
        
        for nh_filename, filename in zip(non_handsign_filenames, filenames):
            tmp_fname = np.genfromtxt(filename+".txt")
            
            if 1.0 not in tmp_fname[:,0]:
                nh_fname = np.genfromtxt(nh_filename)
                #tmp_fname = np.append(tmp_fname, nh_fname, axis=0)
                with open(filename+".txt", "ab") as f:
                    np.savetxt(f, nh_fname, fmt='%s')

        # Send dataset to be trained on server
        server_ip = self.configur.get('training', 'server_ip')
        # Server IP and Port to connect
        HOST = server_ip
        PORT = 50007
        # Create a socket connection.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((HOST, PORT))
            BASE_PATH = ""
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
            
            dataset_size = f"{str(len(str_dataset))}_{dataset.shape}_{self.configur.get('training', 'min_loss')}"
            print(dataset_size)
            s.send(dataset_size.encode())
            s.send(str_dataset.encode())
            print ('Dataset sent to server')
                

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
            MODELS_PATH = fr'{self.BASE_PATH}models'
            if not os.path.exists(MODELS_PATH):
                os.mkdir(MODELS_PATH)
            model_filename = f'/lstm_model_{self.username}.pt'

            # Save nn model
            torch.save(model, MODELS_PATH+model_filename)
        except socket.error as err:
            print(f"Server error : {err}")
            print("Start training using this PC instead")
            model = train_dataset(self.username)
            # Save nn model
            #torch.save(model, MODELS_PATH+model_filename)
        
        cv2.destroyAllWindows()
        

    # mouse callback function
    def line_drawing(self, event,x,y,flags,param):

        if event==cv2.EVENT_LBUTTONDOWN:
            self.start_drawing=True
            self.drawing=True
            self.pt1_x, self.pt1_y = x,y
            self.start_time = time.time()

        elif event==cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_drawing:
                #print("EVENT_MOUSEMOVE")
                cv2.line(self.img,(self.pt1_x, self.pt1_y),(x,y),color=(255,255,255),thickness=3)
                self.pt1_x, self.pt1_y=x,y
        elif event==cv2.EVENT_LBUTTONUP:
            if self.start_drawing:
                self.drawing=False
                cv2.line(self.img,(self.pt1_x, self.pt1_y),(x,y),color=(255,255,255),thickness=3)

        if event==cv2.EVENT_RBUTTONDOWN:
            #print("EVENT_RBUTTONDOWN")
            pass


    def writeCsv(self):
        data_xys = [np.array([]), np.array([])]
        final_datas = [self.interp_x, self.interp_y]
        print(f"username : {self.username}")

        base_path = f"{self.BASE_PATH}dataset/{self.username}"
        os.makedirs(base_path, exist_ok=True)
        
        filenames = [fr"{base_path}/axis_x_train",
                     fr"{base_path}/axis_y_train"]

         
        # Csv header / dataset label
        #dataset_label = ["id"]
        dataset_label = []
        for i in range(0,self.i_range):
            dataset_label.append(f"attr{i}")
        dataset_label.append("target")
        

        for filename, f_data, data_xy in zip(filenames, final_datas, data_xys):
            fle = Path(filename+".csv")
            fle.touch(exist_ok=True)
            #f = open(fle)
            
            temp_df = pd.read_csv(filename+".csv", sep=',', names=dataset_label, header=None)
            df = pd.DataFrame(temp_df.values, columns=dataset_label)

            # Insert Header to csv if csv still empty
            if df.empty:
                df.to_csv(filename+".csv", mode='a', index=False, header=dataset_label, sep=',')
                
            final_data = f_data
                
            # Append Label at the end
            # Change this to change the Target Label
            final_data = np.append(final_data, [2])

            # Save data to csv
            dataset = dict(zip(dataset_label, final_data))
            df = pd.DataFrame(dataset, index=[0])
            
            df.to_csv(filename+".csv", mode='a', index=False, header=False)

            # Format data to be read by Neural Network Program
            data_xy = df.to_numpy()

            data_xy = np.array([np.roll(row, 1) for row in data_xy])
                
            with open(filename+".txt", "ab") as f:
                np.savetxt(f, data_xy, fmt='%s')


class CreateNonHandsignDataset:
    def __init__(self):
        self.configur = ConfigParser()
        print(self.configur.read(f'{self.BASE_PATH}config.ini'))

        self.i_range = self.configur.getint('dataset', 'i_range')

        self.show_hint = True
        self.start_drawing = False # true if left mouse is pressed until reach i_range
        self.drawing = False # true if left mouse is pressed down
        self.pt1_x , self.pt1_y = None , None
        self.start_time = 0

        self.array_x, self.array_y = np.array([]), np.array([])
        self.interp_x, self.interp_y = np.array([]), np.array([])
        self.img = None
    
    def run(self):
        self.show_hint = True
        self.start_drawing = False
        self.array_x, self.array_y = np.array([]), np.array([])
        
        # Fullscreen
        screen = get_monitors()[0]
        screen_width, screen_height = screen.width, screen.height
        self.img = np.zeros((screen_height,screen_width,3), np.uint8)
        cv2.putText(self.img, f"Draw non handsign / Press esc to close", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Fullscreen
        cv2.namedWindow('img', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        cv2.setMouseCallback('img',self.line_drawing)
        # Getting the height and width of the image
        win_width = self.img.shape[1]
        win_height = self.img.shape[0]
        print(f"scren size : {win_width}, {win_height}")

        half_height = int(win_height/2)
        half_width = int(win_width/2)
        cv2.line(self.img, (0, int(half_height)), (win_width, half_height), (0, 0, 255), 1)
        cv2.line(self.img, (half_width, 0), (half_width, win_height), (0, 0, 255), 1)

        measure1 = time.time()
        measure2 = time.time()
        count = 0
        while(1):
            if self.start_drawing == True:
                if self.show_hint == True:
                    self.img = np.zeros((screen_height,screen_width,3), np.uint8)
                    cv2.line(self.img, (0, int(half_height)), (win_width, half_height), (0, 0, 255), 1)
                    cv2.line(self.img, (half_width, 0), (half_width, win_height), (0, 0, 255), 1)
                    self.show_hint = False
                if measure2 - measure1 >= 0.01:
                    count += 1
                    self.array_x = np.append(self.array_x, self.pt1_x)
                    self.array_y = np.append(self.array_y, self.pt1_y)

                    measure1 = measure2
                    measure2 = time.time()
                else:
                    measure2 = time.time()

                if count >= self.i_range:
                    self.drawing = False
                    self.start_drawing=False
                    end_time = time.time()
                    print(f"Elapsed Time : {timedelta(seconds=end_time - self.start_time)}")
                    # Scale axis to (-2, 2) coordinates
                    self.interp_x = np.interp(self.array_x,[np.amin(self.array_x),np.amax(self.array_x)],[-2,2])
                    self.interp_y = np.interp(self.array_y,[np.amin(self.array_y),np.amax(self.array_y)],[2,-2])

                    # Call Write Data to Csv Function
                    self.writeCsv()
                    
                    break
            
            cv2.imshow('img',self.img)
            if cv2.waitKey(1) & 0xFF == 27:
                self.is_esc = True
                break
        cv2.destroyAllWindows()
        

    # mouse callback function
    def line_drawing(self, event,x,y,flags,param):

        if event==cv2.EVENT_LBUTTONDOWN:
            self.start_drawing=True
            self.drawing=True
            self.pt1_x, self.pt1_y = x,y
            self.start_time = time.time()

        elif event==cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_drawing:
                #print("EVENT_MOUSEMOVE")
                cv2.line(self.img,(self.pt1_x, self.pt1_y),(x,y),color=(255,255,255),thickness=3)
                self.pt1_x, self.pt1_y=x,y
        elif event==cv2.EVENT_LBUTTONUP:
            if self.start_drawing:
                self.drawing=False
                cv2.line(self.img,(self.pt1_x, self.pt1_y),(x,y),color=(255,255,255),thickness=3)

        if event==cv2.EVENT_RBUTTONDOWN:
            #print("EVENT_RBUTTONDOWN")
            pass


    def writeCsv(self):
        final_datas = [self.interp_x, self.interp_y]

        base_path = f"{self.BASE_PATH}dataset"
        os.makedirs(base_path, exist_ok=True)
        
        filenames = [fr"{base_path}/non_handsign_x",
                     fr"{base_path}/non_handsign_y"]


        for filename, final_data in zip(filenames, final_datas):
            # Append Label at the end
            # Change this to change the Target Label
            final_data = np.append(final_data, [1])
            final_data = np.atleast_2d(final_data)

            final_data = np.array([np.roll(row, 1) for row in final_data])
                
            with open(filename+".txt", "ab") as f:
                np.savetxt(f, final_data, fmt='%s')


# Run only if in this namespace
if __name__ == '__main__':
##    test = CreateDataset('test1')
##    test.run()
    test = CreateNonHandsignDataset()
    test.run()
