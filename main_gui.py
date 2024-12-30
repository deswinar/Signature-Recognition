import sys
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QMainWindow, QMessageBox, QInputDialog, QLineEdit
from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QIntValidator, QDoubleValidator

from ui.main_ui import Ui_MainWindow

import re

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import os
import shutil

from models import ConvLSTMNet

#from train_data_lstm import train_dataset
from create_dataset import CreateDataset, CreateNonHandsignDataset
from predict_data_lstm import Predict
#import train_data_lstm as train

from configparser import ConfigParser

class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # Base path to main folder
        # Raspberry
        self.BASE_PATH = r"/home/sulthon/Downloads/HandsignRecognition/"
        # Others
        self.BASE_PATH = ""
        
        self.set_validator()

        self.configur = ConfigParser()
        print(self.configur.read(f'{self.BASE_PATH}config.ini'))
        self.init_setting()

        self.password = self.configur.get('admin', 'password')

        self.is_admin = False

        # Register / Check In
        self.registerPushButton.clicked.connect(self.register)
        self.checkInPushButton.clicked.connect(self.predict)

        # Admin
        self.tabWidget.tabBarClicked.connect(self.tab_widget_clicked)
        self.createNHDPushButton.clicked.connect(self.create_non_handsign)
        self.removeLastNHDPushButton.clicked.connect(self.remove_last_non_handsign)
        self.removeAllNHDPushButton.clicked.connect(self.remove_all_non_handsign)

        self.removeUserPushButton.clicked.connect(self.remove_user)
        self.removeAllUserPushButton.clicked.connect(self.remove_all_user)

        self.saveSettingPushButton.clicked.connect(self.save_setting)

        # Admin Utilities
        self.exitPushButton.clicked.connect(self.exit_app)
        self.shutdownPushButton.clicked.connect(self.shutdown_device)
        self.restartPushButton.clicked.connect(self.restart_device)

        # Fullscreen
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.showMaximized()

    def init_setting(self):
        self.checkInUsernameLineEdit.setEnabled(False)
        
        self.i_rangeLineEdit.setText(self.configur.get('dataset', 'i_range'))
        self.i_dataLineEdit.setText(self.configur.get('dataset', 'i_data'))
        self.minLossLineEdit.setText(self.configur.get('training', 'min_loss'))
        self.bestConfLineEdit.setText(self.configur.get('training', 'best_conf'))
        self.serverIPLineEdit.setText(self.configur.get('training', 'server_ip'))
        self.checkInUsernameLineEdit.setText(self.configur.get('dataset', 'current_user'))
        self.currentUserLineEdit.setText(self.configur.get('dataset', 'current_user'))

    def set_validator(self):
        int_validator = QIntValidator(1, 9999, self)
        double_validator = QDoubleValidator(0.00001, 9.99999, 5)
        
        self.i_rangeLineEdit.setValidator(int_validator)
        self.i_dataLineEdit.setValidator(int_validator)
        self.minLossLineEdit.setValidator(double_validator)
        self.bestConfLineEdit.setValidator(double_validator)

    def tab_widget_clicked(self, index):
        if index == 2:
            if not self.is_admin:
                password, ok = QInputDialog.getText(self, 'Request Admin Access',
                                                    'Enter admin password:',
                                                    QLineEdit.EchoMode.Password)
                if password == self.password and ok:
                    self.is_admin = True
                    self.groupBox_0.setEnabled(True)
                    self.groupBox_1.setEnabled(True)
                    self.groupBox_2.setEnabled(True)
                else:
                    self.is_admin = False
                    self.groupBox_0.setEnabled(False)
                    self.groupBox_1.setEnabled(False)
                    self.groupBox_2.setEnabled(False)

    def create_non_handsign(self):
        create_dataset = CreateNonHandsignDataset()
        create_dataset.run()

    def remove_last_non_handsign(self):
        print("Removing last non handsign dataset..")
        file_paths = [fr'{self.BASE_PATH}dataset/non_handsign_x.txt',
                      fr'{self.BASE_PATH}dataset/non_handsign_y.txt']
        
        if os.path.exists(file_paths[0]):
            for fpath in file_paths:
                dataset = np.genfromtxt(fpath)
                dataset = dataset[:-1]
                np.savetxt(fpath, dataset, fmt='%s')
            QMessageBox.about(self, "Info", "Last non handsign dataset removed")
        else:
            print("Can not delete the file as it doesn't exists")
            QMessageBox.critical(self, "Warning", "Non handsign dataset doesn't exist")

    def remove_all_non_handsign(self):
        print("Removing all non handsign dataset..")
        file_paths = [fr'{self.BASE_PATH}dataset/non_handsign_x.txt',
                      fr'{self.BASE_PATH}dataset/non_handsign_y.txt']
        
        if os.path.exists(file_paths[0]):
            for fpath in file_paths:
                os.remove(fpath)
            QMessageBox.about(self, "Info", "Non handsign dataset cleared")
        else:
            print("Can not delete the file as it doesn't exists")
            QMessageBox.critical(self, "Warning", "Non handsign dataset already cleared")

    def remove_user(self):
        username = self.usernameLineEdit.text()
        print(f'username : {username}')
        if self.is_input_empty([username]):
            self.usernameLineEdit.setPlaceholderText("Required!!")
            return

        # Directory name
        parent = fr"{self.BASE_PATH}dataset/"
        directory = f"{username}"
         
        # Path
        path = os.path.join(parent, directory)
        if not os.path.isdir(path):
            QMessageBox.warning(self, "Warning", f"Username: {username} is not exist")
            return
         
        try:
            shutil.rmtree(path)
            QMessageBox.about(self, "Info", "User removed")
        except OSError as err:
            print(err)
            QMessageBox.critical(self, "Error", f"{err}")

    def remove_all_user(self):
        exist = False
        parent = fr'{self.BASE_PATH}dataset'
        directories = next(os.walk(parent))[1]
        if len(directories) > 0: exist = True
        try:
            for i, directory in enumerate(directories):
                shutil.rmtree(f"{parent}/{directory}")
                print(f"{parent}/{directory}")
            if exist:
                QMessageBox.about(self, "Info", "All user removed")
            else:
                QMessageBox.warning(self, "Warning", "No user exist")
        except OSError as err:
            print(err)
            QMessageBox.critical(self, "Error", f"{err}")

    def save_setting(self):
        i_range = self.i_rangeLineEdit.text()
        i_data = self.i_dataLineEdit.text()
        min_loss = self.minLossLineEdit.text()
        best_conf = self.bestConfLineEdit.text()
        server_ip = self.serverIPLineEdit.text()
        current_user = self.currentUserLineEdit.text()

        if self.is_input_empty([i_range, i_data, min_loss, best_conf]):
            self.i_rangeLineEdit.setPlaceholderText("Required!!")
            self.i_dataLineEdit.setPlaceholderText("Required!!")
            self.minLossLineEdit.setPlaceholderText("Required!!")
            self.bestConfLineEdit.setPlaceholderText("Required!!")
            self.currentUserLineEdit.setPlaceholderText("Required!!")
            return

        self.configur.set('dataset', 'i_range', i_range)
        self.configur.set('dataset', 'i_data', i_data)
        self.configur.set('training', 'min_loss', min_loss)
        self.configur.set('training', 'best_conf', best_conf)
        self.configur.set('training', 'server_ip', server_ip if len(server_ip) > 0 else '127.0.0.1')
        self.configur.set('dataset', 'current_user', current_user)
        # Writing our configuration file to 'example.cfg'
        with open('config.ini', 'w') as configfile:
            self.configur.write(configfile)
            QMessageBox.about(self, "Info", "Setting saved!")
            self.checkInUsernameLineEdit.setText(self.configur.get('dataset', 'current_user'))
            
    def exit_app(self):
        print('Exitting App...')
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Icon.Question)
        msgBox.setText("Are you sure to exit app?")
        msgBox.setWindowTitle("Exit app")
        msgBox.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

        returnValue = msgBox.exec()
        if returnValue == QMessageBox.StandardButton.Ok:
            sys.exit()

    def shutdown_device(self):
        print('Shutdowning Device...')
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Icon.Question)
        msgBox.setText("Are you sure to shutdown?")
        msgBox.setWindowTitle("Shutdown devide")
        msgBox.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

        returnValue = msgBox.exec()
        if returnValue == QMessageBox.StandardButton.Ok:
            os.system("sudo shutdown -h now")

    def restart_device(self):
        print('Restarting Device...')
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Icon.Question)
        msgBox.setText("Are you sure to restart?")
        msgBox.setWindowTitle("Restart devide")
        msgBox.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

        returnValue = msgBox.exec()
        if returnValue == QMessageBox.StandardButton.Ok:
            os.system("sudo reboot -h now")

    def register(self):
        self.drawing = True
        print("Register Started..")
        username = self.registerUsernameLineEdit.text()
        print(f'username : {username}')
        if self.is_input_empty([username]):
            self.registerUsernameLineEdit.setPlaceholderText("Required!!")
            return

        path = fr'{self.BASE_PATH}dataset/{username}/'
        if os.path.isdir(path):
            QMessageBox.warning(self, "Warning", f"Username: {username} is already exist")
            return
        
        create_dataset = CreateDataset(username)
        create_dataset.run()


    def predict(self):
        print("Check In Started..")
        username = self.checkInUsernameLineEdit.text()
        print(f'username : {username}')
        if self.is_input_empty([username]):
            self.checkInUsernameLineEdit.setPlaceholderText("Required!!")
            return
        
        path = fr'{self.BASE_PATH}dataset/{username}/'
        if not os.path.isdir(path):
            QMessageBox.warning(self, "Warning", f"Username: {username} is not exist")
            return
        predict = Predict(username)
        predict.run()

    def is_input_empty(self, list_input):
        def is_empty_or_blank(msg):
            return re.search("^\s*$", msg)
        
        result = any([is_empty_or_blank(elem) for elem in list_input])
        if result:
           #print('List input contains empty string')
           return True
        else :
           #print('List input does not contains any empty string')
           return False
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())
