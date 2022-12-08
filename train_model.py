from tkinter import *
import cv2
import os
import numpy as np
from PIL import Image
import xlsxwriter as xl
from datetime import date




###For take images for datasets
def take_img(roll_num):

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    sampleNum = 0
    user_id = +1
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # incrementing sample number
            sampleNum = sampleNum + 1
            # saving the captured face in the dataset folder + '.' + str(sampleNum)
            #cv2.imwrite("dataset/ " + roll_num + '.'+ str(user_id)+'.' + str(sampleNum)+".jpg",
            #                    gray[y:y + h, x:x + w])
            cv2.imwrite("dataset/ " + roll_num + '.'+ roll_num[-2]+roll_num[-1]+'.' + str(sampleNum)+".jpg",
            gray[y:y + h, x:x + w])
            cv2.imshow('Frame', img)
                # wait for 100 miliseconds
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                # break if the sample number is morethan 100
        elif sampleNum > 500:
            print(f"** YOU ARE ALLOTED USER ID IS {roll_num[-2]+roll_num[-1]}")
            print("Registration is successfull!!")
            return user_id
    cam.release()
    cv2.destroyAllWindows()


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    #print(imagePaths)
    faceSamples = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image

        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)
        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
        #print(Ids)
    return faceSamples, Ids
      


###For train the model
def trainimg():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    global detector
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    try:
        global faces,Id
        faces, Id = getImagesAndLabels("dataset")
        #print(faces)
    except Exception as e:
        print("please make a folder and put images")

    recognizer.train(faces, np.array(Id)) 
    try:
        recognizer.save("model/trained_model2.yml")
    except Exception as e:
        print("Please make model folder")
    print("Model data Trained!")
	
	




def registration(cv_sheet,row_index):
	####
	#print(row_index)
	#cv_sheet = reg_data.active
    #user_id = 0
    roll_num = input("Enter Your Roll Number: ")
    name = input("Enter your name: ")
    sem_num = int(input("Enter Your current semister: "))
	##User details -> excel sheet
	#cv_sheet[f'A{row_index}']=roll_num
    cv_sheet.write(f'A{row_index}',roll_num)
    cv_sheet.write(f'B{row_index}',name)
    cv_sheet.write(f'C{row_index}',sem_num)
    clct = input("Enter y to give sample images:")
    if clct =='Y' or clct == 'y':
        val  = take_img(roll_num)
        cv_sheet.write('D'+str(row_index),"Collected")

    else:
        print("You need to give sample images to recognise!!!")
    #if val == 0:
    #    cv_sheet.write('D'+str(row_index),"Present")
    #else:
    #    cv_sheet.write('D'+str(row_index),'Absent')
    row_index += 1
	##
	#reg_data.save('regis.xlsx')
    add = input("Enter Y to add Student else N: ")
    if add == 'Y' or add == 'y':
        #user_id += 1
        registration(cv_sheet,row_index)
    else:
        trainimg()

def crte_sheet():
    curnt_date = date.today()
    row_index = 2
    reg_data = xl.Workbook("regis.xlsx")
    cv_sheet = reg_data.add_worksheet("computer vision")
    cv_sheet.write('A1','Roll number')
    cv_sheet.write('B1','Name')
    cv_sheet.write('C1','Semister')
    cv_sheet.write('D1','Samples')
    cv_sheet.write('E1',f'{curnt_date}')
    #user_id = 0
    registration(cv_sheet,row_index)
    reg_data.close()

print("Hello User WELCOME!!\n \nPlease Choose Your option:\n")
choice = 0
choice = int(input("1.Student registration \n2.Authentication Process \n3.Train data:"))

if choice == 1:
	res = crte_sheet()
elif choice == 3:
    res = trainimg()
	#reg_data.save('regis.xlsx')
	#reg_data.close()
else:
	print("\nIncorrect Choice!!!\n")
	



	
	
