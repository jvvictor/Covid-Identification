
'''
EDA - Exploratory Data Analysis

'''

from enum import unique
from re import search
import numpy
import pandas as pd
import os
import glob
import cv2
from sklearn.model_selection import train_test_split


class Preprocess:  

    def __init__(self, data_path):

        self.data_path = data_path

        self.metadata = self.create_data_frame()

        return


    def create_data_frame(self):

        data_frame = pd.read_csv(self.data_path, sep=",", header=0)

        return data_frame

    def Filter(self):

        for n in range(self.metadata.shape[0]):
             if self.metadata['modality'][n] != "X-ray":
                 self.metadata = self.metadata.drop([n], axis=0)
    
        self.metadata.reset_index(drop = True, inplace = True)

        for j in range(self.metadata.shape[0]):
            if search('.png', self.metadata['filename'][j]):
                pass
            elif search('.jpg', self.metadata['filename'][j]):
                pass
            elif search('.jpeg', self.metadata['filename'][j]):
                pass
            elif search('.JPG', self.metadata['filename'][j]): 
                pass
            else:
                self.metadata = self.metadata.drop([j], axis=0)

        self.metadata.reset_index(drop = True, inplace = True)

        return

    def unique_id(self):

        listpatient = [self.metadata['patientid'][0]]

        for p in range(self.metadata.shape[0]):
             if self.metadata['patientid'][p] not in listpatient:
                 listpatient.append(self.metadata['patientid'][p])

        print("")
        print("The number of patients is:", len(listpatient))
        print("")


        return

    def unique_dis(self):

        listdis = [self.metadata['finding'][0]]

        for k in range(self.metadata.shape[0]):
         if self.metadata['finding'][k] not in listdis:
             listdis.append(self.metadata['finding'][k])

        print("")
        print("The diseases are:", listdis )
        print("The number os diseases is:", len(listdis))
        print("")

        return

    def show_image(self, number_images):

        cv2.namedWindow("T", 0)

        for l in range(number_images):
            img = cv2.imread(os.path.join('../images', self.metadata['filename'][l]), 1)
            cv2.imshow('T', img)
            cv2.waitKey(0)

        return

    def transf_binary(self):

        for j in range(self.metadata.shape[0]):
             if self.metadata['finding'][j] != "COVID-19":
                    self.metadata['finding'][j] = 0
             else:
                    self.metadata['finding'][j] = 1

        return

    def split_train_data(self):

        unique_patients = self.metadata['patientid'].unique()
        train , test = train_test_split(unique_patients, test_size = 0.05, random_state = 0, shuffle = True)

        self.traindata = self.metadata.copy()
        self.valdata = self.metadata.copy()

        for i in range(self.metadata.shape[0]):
            if self.metadata['patientid'][i] not in test:
                    self.traindata = self.traindata.drop([i], axis = 0)
            else:
                    self.valdata = self.valdata.drop([i], axis = 0)

        self.traindata.reset_index(drop = True, inplace = True)
        self.valdata.reset_index(drop = True, inplace = True)

        return











if __name__ == "__main__":

    data_path = "C:/Users/jvsoa/Desktop/COVID/metadata.csv"
    covid_1 = Preprocess(data_path)
    covid_1.Filter()
    covid_1.unique_dis()
    covid_1.unique_id()
    covid_1.show_image(int(input("How many images to show?: ")))
    covid_1.transf_binary()
    covid_1.split_train_data()

    print(covid_1.traindata.head(5))
    print(covid_1.valdata.head(5))
    
    '''
    metadata = pd.read_csv("../metadata.csv", header=0)
    for n in range(metadata.shape[0]):
        if metadata['modality'][n] != "X-ray":
            metadata = metadata.drop([n], axis=0)
    
    metadata.reset_index(drop = True, inplace = True)
    
    for j in range(metadata.shape[0]):
        if search('.png', metadata['filename'][j]):
            pass
        elif search('.jpg', metadata['filename'][j]):
            pass
        elif search('.jpeg', metadata['filename'][j]):
            pass
        elif search('.JPG', metadata['filename'][j]): 
            pass
        else:
            metadata = metadata.drop([j], axis=0)

    metadata.reset_index(drop = True, inplace = True)

    want = ["patientid","finding","modality","filename"]
    metadata = metadata[want]
 
    listdis = [metadata['finding'][0]]

    for k in range(metadata.shape[0]):
        if metadata['finding'][k] not in listdis:
            listdis.append(metadata['finding'][k])
    
    print("")
    print("The diseases are:", listdis )
    print("The number os diseases is:", len(listdis))

    listpatient = [metadata['patientid'][0]]

    for p in range(metadata.shape[0]):
        if metadata['patientid'][p] not in listpatient:
            listpatient.append(metadata['patientid'][p])

    print("")
    print("The number of patients is:", len(listpatient))

    cv2.namedWindow("T", 0)

    for l in range(5):
        img = cv2.imread(os.path.join('../images', metadata['filename'][l]), 1)
        cv2.imshow('T', img)
        cv2.waitKey(0)

    for j in range(metadata.shape[0]):
        if metadata['finding'][j] != "COVID-19":
            metadata['finding'][j] = 0
        else:
            metadata['finding'][j] = 1

    unique_patients = metadata['patientid'].unique()
    train , test = train_test_split(unique_patients, test_size = 0.05, random_state = 0, shuffle = True)

    traindata = metadata.copy()
    valdata = metadata.copy()

    for i in range(metadata.shape[0]):
        if metadata['patientid'][i] not in test:
            traindata = traindata.drop([i], axis = 0)
        else:
            valdata = valdata.drop([i], axis = 0)

    traindata.reset_index(drop = True, inplace = True)
    valdata.reset_index(drop = True, inplace = True)

    print("")
    print("Work complete")

    '''