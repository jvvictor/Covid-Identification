import numpy as np
import cv2


from data_loader import *
from eda import *

if __name__ == '__main__':  

    data_path = "C:/Users/jvsoa/Desktop/COVID/metadata.csv"
    demo_1 = preprocess(data_path)
    demo_1.filter()

    #covid_1.unique_dis()       #Show unique diseases
    #covid_1.unique_id()        #Count unique patient
    #covid_1.show_image(int(input("How many images to show?: "))) #Show some images

    demo_1.transf_binary()
    demo_1.split_train_data()

    data_loader_demo_1 = DataLoader(demo_1.train_data, demo_1.val_data, "C:/Users/jvsoa/Desktop/COVID/images") 
    data_loader_demo_1.CreateImageGenerator()
    data_loader_demo_1.CreateDataLoader()

    image, label = data_loader_demo_1.train_data_loader.__getitem__(2)
    cv2.namedWindow("T", 0)
    cv2.imshow("T", image[0])
    cv2.waitKey(0)
    print(label)
