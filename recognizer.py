import cv2
import Pedestrian_detection as pdf
#pip install opencv-contrib-python
def recog(lst):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('model/trained_model2.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX

    #cam = cv2.VideoCapture(0)
    while True:
        #ret, im =image.read()
        gray=cv2.cvtColor(lst[0],cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)
        
        for(x,y,w,h) in faces:
            roll_num, conf = recognizer.predict(gray[y:y+h,x:x+w])
            print(roll_num)
            #cv2.rectangle(lst[0], (x, y), (x + w, y + h), (0, 260, 0), 7)
            cv2.rectangle(lst[0],(lst[1],lst[3]),(lst[2],lst[4]),(0,255,0),7)
            #cv2.putText(im, str(roll_num), (x,y-40),font, 2, (255,255,255), 3)
            cv2.putText(lst[0], str(roll_num), (x,y-40),font, 2, (255,255,255), 3)
            
        cv2.imshow('im',lst[0])
        #if cv2.waitKey(10) & 0xFF==ord('q'):
        #    break
        #else:
        #    pdf.pd_detect()


        image = pdf.pd_detect()
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
        recog(image)
    #cam.release()
    cv2.destroyAllWindows()

print("Enter R to start recognition: ")
choice = input()
if choice == 'R' or choice == 'r':
    #recog()
    lst = pdf.pd_detect()
    recog(lst)

