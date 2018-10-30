import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sns
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
from sklearn.metrics import classification_report



def lineUpData(raw_data, annotated_data) :
    data = pd.read_csv(raw_data)
    labeled_data =  pd.read_csv(annotated_data)

    diff = len(data) - len(labeled_data)
    print("diff is: ", diff)
    if diff > 0:
        data = data[:-diff]
    if diff < 0:
        diff = diff * -1
        labeled_data = labeled_data[:-diff]
    cleaned_data = data[data[' confidence'] > .70] 
    removed_data = data[data[' confidence'] <= .70] 

    #align the data
    aligned_labels = []

    newFrames = []
    for index, row in removed_data.iterrows():
        newFrames.append(int(row["frame"]) - 1)
    labeled_data = labeled_data.drop(newFrames)
    return cleaned_data, labeled_data

    
def main():
    cleaned_data1, labeled_data1 = lineUpData("webcam_test_2017-09-30-12-26-14.csv", "annos_s17_2017-09-30_3.csv")
    print(len(cleaned_data1))
    print(len(labeled_data1))
    cleaned_data2, labeled_data2 = lineUpData("webcam_test_2017-09-30-12-05-22.csv", "annos_s17_2017-09-30_2.csv")
    print(len(cleaned_data2))
    print(len(labeled_data2))
    cleaned_data3, labeled_data3 = lineUpData("webcam_test_2017-09-30-11-56-56.csv", "annos_s17_2017-09-30_1.csv")
    print(len(cleaned_data3))
    print(len(labeled_data3))
    cleaned_data4, labeled_data4 = lineUpData("webcam_test_2017-09-26-19-32-41.csv", "annos_s16_2017-09-26.csv")
    print(len(cleaned_data4))
    print(len(labeled_data4))
    cleaned_data5, labeled_data5 = lineUpData("webcam_test_2017-09-22-19-39-14.csv", "annos_s14_2017-09-22.csv")
    print(len(cleaned_data5))
    print(len(labeled_data5))
    
    cleaned_data = pd.DataFrame()
    cleaned_data = cleaned_data.append(cleaned_data1, ignore_index = True)  
    cleaned_data = cleaned_data.append(cleaned_data2, ignore_index = True)
    cleaned_data = cleaned_data.append(cleaned_data3, ignore_index = True)
    cleaned_data = cleaned_data.append(cleaned_data4, ignore_index = True)
    # cleaned_data = cleaned_data.append(cleaned_data5, ignore_index = True)

    labeled_data = pd.DataFrame()
    labeled_data = labeled_data.append(labeled_data1, ignore_index = True)  
    labeled_data = labeled_data.append(labeled_data2, ignore_index = True)
    labeled_data = labeled_data.append(labeled_data3, ignore_index = True)
    labeled_data = labeled_data.append(labeled_data4, ignore_index = True)
    # labeled_data = labeled_data.append(labeled_data5, ignore_index = True)

    print(cleaned_data.shape)
    print(labeled_data.shape)

    labels = labeled_data["C_Gaze"]

    labels_train = []
    labels_test = []
    for i in range(len(labels)):
        if labels.values[i] == "away":
            labels_train.append(False)
        else: 
            labels_train.append(True)

    
    labels = labeled_data5["C_Gaze"]
    for i in range(len(labels)):
        if labels.values[i] == "away":
            labels_test.append(False)
        else: 
            labels_test.append(True)

    cleaned_data_test = cleaned_data5
    cleaned_data_train = cleaned_data

    LogReg = LogisticRegression(max_iter=10000, penalty='l2', tol=.0001, class_weight="balanced")

    LogReg.fit(cleaned_data_train, labels_train)
    y_pred = LogReg.predict(cleaned_data_test)
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(labels_test, y_pred)
    print(confusion_matrix)


main()
    

