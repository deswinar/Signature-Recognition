import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models import LSTMNet, ConvLSTMNet, ConvNet
import models

import time
from datetime import timedelta, datetime
import csv
import os.path
from pathlib import Path
import sys
import ctypes
from screeninfo import get_monitors

from configparser import ConfigParser
import logging

class Predict:
    def __init__(self, username):
        self.username = username

        # Base path to main folder
        # Raspberry
        self.BASE_PATH = r"/home/sulthon/Downloads/HandsignRecognition/"
        # Others
        self.BASE_PATH = ""

        logging.basicConfig(filename=f"{self.BASE_PATH}predictions/predictions.log",
                            level=logging.INFO,
                            format="%(asctime)s %(message)s")

        self.configur = ConfigParser()
        print(self.configur.read(f'{self.BASE_PATH}config.ini'))

        self.i_range = self.configur.getint('dataset', 'i_range')

        #self.i_range = 200 # Range of how many times axis will be recorded

        self.is_unlocked = False

        self.state_drawing = 0 # 0:not finish, 1:finish, 2:finish close
        self.start_drawing = False # true if left mouse is pressed until reach i_range
        self.drawing = False # true if left mouse is pressed down
        self.pt1_x , self.pt1_y = None , None
        self.start_time = 0

        self.array_x = np.array([])
        self.array_y = np.array([])

    def run(self):
        # Full Screen
##        user32 = ctypes.windll.user32
##        screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        screen = get_monitors()[0]
        screen_width, screen_height = screen.width, screen.height
        self.img = np.zeros((screen_height,screen_width,3), np.uint8)

        # Full Screen
        cv2.namedWindow('img', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        cv2.setMouseCallback('img',self.line_drawing)
        # Getting the height and width of the image
        screen_width = self.img.shape[1]
        screen_height = self.img.shape[0]
        print("Screen Size : ", screen_width, screen_height)

        half_height = int(screen_height/2)
        half_width = int(screen_width/2)
        # Draw axis line
        cv2.line(self.img, (0, int(half_height)), (screen_width, half_height), (0, 0, 255), 1)
        cv2.line(self.img, (half_width, 0), (half_width, screen_height), (0, 0, 255), 1)

        measure1 = time.time()
        measure2 = time.time()
        count = 0
        while(1):
            if self.start_drawing == True and self.state_drawing == 0:
                if measure2 - measure1 >= 0.01:
                    count += 1
                    self.array_x = np.append(self.array_x, self.pt1_x)
                    self.array_y = np.append(self.array_y, self.pt1_y)
    ##                print(pt1_x,pt1_y, count)
                    measure1 = measure2
                    measure2 = time.time()
                else:
                    measure2 = time.time()

                if count >= self.i_range:
                    #cv2.setMouseCallback('img', lambda *args : None)
                    self.start_drawing = False
                    self.state_drawing = 1
                    end_time = time.time()
                    print("Drawing Elapsed Time : ", timedelta(seconds=end_time - self.start_time))
                    # Scale axis to (-2, 2) coordinates
                    self.interp_x = np.interp(self.array_x,[np.amin(self.array_x),np.amax(self.array_x)],[-2,2])
                    self.interp_y = np.interp(self.array_y,[np.amin(self.array_y),np.amax(self.array_y)],[2,-2])
##                    print("x = ", array_x)
##                    print("y = ", array_y)
##                    print("--------------------------------")
##                    print("intx = ", interp_x)
##                    print("inty = ", interp_y)

                    # Call Write Data to Txt Function
                    file_to_predicts = self.write_predict_file()

                    # Predict Label
                    conf, classes = self.predict_data_lstm(file_to_predicts)
                    torch.set_printoptions(sci_mode=False)
                    print(conf, classes)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    #cv2.putText(self.img, f"Truth Label : {[argv[1]] if len(argv)>1 else [1]}", (10,30), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    logtext = f"checkin; username:{self.username}; "
                    for i, (cf, cl)  in enumerate(zip(conf[0], classes[0])):
                        loc_y = 50 + (i*20)
                        cv2.putText(self.img, f"Predicted : {cl+1} with conf : {cf}", (10,loc_y), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        logtext += f"(label:{cl+1},conf:{cf}); "
                        if i == 0 and int(cl+1) == 2 and cf > 0.85:
                            print(f"Locker Unlocked.")
                            self.is_unlocked = True
                            # Run GPIO Unlock here...
                    logging.info(logtext)
                    
                    if self.is_unlocked:
                        cv2.putText(self.img, f"Good match", (10,90), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
##                        break
                    else:
                        cv2.putText(self.img, f"Bad match", (10,90), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    
            if self.state_drawing == 2:
                self.is_unlocked = False
                # Run GPIO Lock here...
                break
            cv2.imshow('img',self.img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    # mouse callback function
    def line_drawing(self, event,x,y,flags,param):

        if event==cv2.EVENT_LBUTTONDOWN:
            self.start_drawing=True
            self.drawing=True
            self.pt1_x,self.pt1_y = x,y
            self.start_time = time.time()
            if self.state_drawing == 1:
                self.state_drawing = 2

        elif event==cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_drawing:
                cv2.line(self.img,(self.pt1_x,self.pt1_y),(x,y),color=(255,255,255),thickness=3)
                self.pt1_x,self.pt1_y=x,y
        elif event==cv2.EVENT_LBUTTONUP:
            if self.start_drawing:
                self.drawing=False
                cv2.line(self.img,(self.pt1_x,self.pt1_y),(x,y),color=(255,255,255),thickness=3)

        if event==cv2.EVENT_RBUTTONDOWN:
            #print(len(argv))
            if self.state_drawing == 1:
                self.state_drawing = 2
            pass

    def write_predict_file(self):
        data_xys = [np.array([]), np.array([])]
        final_datas = [self.interp_x, self.interp_y]

        BASE_PATH = "predictions"
        Path(BASE_PATH).mkdir(parents=True, exist_ok=True)

        datetime_now = datetime.now().strftime("%d%m%Y_%H%M%S")
        
        filenames = [fr"{self.BASE_PATH}{BASE_PATH}/predict_x_{datetime_now}",
                     fr"{self.BASE_PATH}{BASE_PATH}/predict_y_{datetime_now}"]

        file_to_predicts = []
         
        for filename, f_data, data_xy in zip(filenames, final_datas, data_xys):
            final_data = f_data

            # Append Label at the first column
            final_data = np.insert(final_data, 0, [1])

            # Format data to be read by Neural Network Program
            df = pd.DataFrame(final_data) 
            
            data_xy = df.to_numpy()
            data_xy = data_xy[:,:]
            data_xy = np.array(data_xy).T
            
            np.savetxt(filename+".txt", data_xy, fmt='%s')
            file_to_predicts.append(filename+".txt")
        return file_to_predicts
            
    def predict_data_lstm(self, file_to_predicts):
        #file_to_predicts = ['dataset/prediction/DUMMY_predict_x.txt', 'dataset/prediction/DUMMY_predict_y.txt']
        device = 'cpu'
        # Get Model Path and Filename
        BASE_PATH = 'models'
        Path(BASE_PATH).mkdir(parents=True, exist_ok=True)
        
        model_filename = fr'{self.BASE_PATH}{BASE_PATH}/lstm_model_{self.username}.pt'
        
        # Load Model and Set to Eval Mode
        model = torch.load(model_filename, map_location=device)
        #print(model)
        model.eval()

        # Load Handsigned Test Data
        data = list()
        for fnp in file_to_predicts:
            data_with_labels = np.genfromtxt(fnp)
            data.append(data_with_labels[1:])
            labels = data_with_labels[0]
            #labels = labels.reshape(1,-1).T
            #print(labels)

        data = np.dstack(data)
        print(f"data.shape : {data.shape}")
        print(f"labels.shape : {labels.shape}")
        
        # Convert test data to tensor
        predict_file = torch.tensor(data)
        
        inputs = predict_file.to(device, dtype=torch.float)
        with torch.no_grad():
            output = model(inputs)

        # Predict the target label
        #probs = torch.nn.functional.softmax(output, dim=1)
        #probs = output
        probs = torch.sigmoid(output)
        top_p, top_class = probs.topk(output[0].shape[0])
        return top_p, top_class
    ##    conf, classes = torch.max(probs, 1)
    ##    return conf, classes

if __name__=="__main__":
    test = Predict('test1')
    test.run()
