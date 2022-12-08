import numpy as np
import cv2
import imutils
import pandas as pd
import time
import xlsxwriter as xl




NMS_THRESHOLD=0.3
MIN_CONFIDENCE=0.2



def pedestrian_detection(image, model, layer_name, personidz=0):
	(H, W) = image.shape[:2]
	results = []


	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	model.setInput(blob)
	layerOutputs = model.forward(layer_name)

	boxes = []
	centroids = []
	confidences = []

	for output in layerOutputs:
		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personidz and confidence > MIN_CONFIDENCE:

				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
	# ensure at least one detection exists
	if len(idzs) > 0:
		# loop over the indexes we are keeping
		for i in idzs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			res = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(res)
	# return the list of results
	return results



labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "weights/yolov4-tiny.weights"
config_path = "cfg/yolov4-tiny.cfg"

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
'''
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
'''

layer_name = model.getLayerNames()
layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]


def pd_detect(cv_sheet,row_index):
	##
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read('model/trained_model2.yml')
	cascadePath = "haarcascade_frontalface_default.xml"
	faceCascade = cv2.CascadeClassifier(cascadePath)
	font = cv2.FONT_HERSHEY_SIMPLEX
	##
	cap = cv2.VideoCapture(0)

	while True:
		(grabbed, image) = cap.read()
		
		if not grabbed:
			break
		image = imutils.resize(image, width=700)
		results = pedestrian_detection(image, model, layer_name,
		personidz=LABELS.index("person"))
		lst = []
		for res in results:
			#cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)
			x1 = int(res[1][0]+(abs(res[1][2]-res[1][0])/4))
			y1 = int(res[1][1])
			x2 = int(res[1][2]-(abs(res[1][2]-res[1][0])/4))
			y2 = int(res[1][1]+(abs(res[1][3]-res[1][1])/8))
			#cv2.rectangle(image, (x1,y1),(x2,y2), (0,0,255), 1)
			lst.append(image)
			lst.append(x1)
			lst.append(x2)
			lst.append(y1)
			lst.append(y2)

			#return lst
		##
		gray=cv2.cvtColor(lst[0],cv2.COLOR_BGR2GRAY)
		faces=faceCascade.detectMultiScale(gray, 1.2,5)
		var = "Unknown"
		count = 0
		for(x,y,w,h) in faces:
			roll_num, conf = recognizer.predict(gray[y:y+h,x:x+w])#(gray[lst[3]:lst[4],lst[1]:lst[2]])#(gray[y:y+h,x:x+w])
			print(conf)
			print(roll_num)
			if conf>90:
				cv2.rectangle(lst[0],(x,y),(x+w, y+h),(0,0,255),3)
				cv2.putText(lst[0], var, (x,y-40),font, 2, (255,255,255), 3)
				count += 1
				print('unkowns : '+ str(count)+'.')
				#cv_sheet.write('D'+str(row_index),"Present")
			else:
				cv2.rectangle(lst[0], (x, y), (x + w, y + h), (0, 255, 0), 3)
				#cv2.rectangle(lst[0],(lst[1],lst[3]),(lst[2],lst[4]),(0,255,0),7)
				#cv2.putText(im, str(roll_num), (x,y-40),font, 2, (255,255,255), 3)
				#cv2.putText(lst[0], str(roll_num), (x,y-40),font, 2, (255,255,255), 3)
				cv2.putText(lst[0], str(roll_num), (x,y-40),font, 2, (255,255,255), 3)
				#
				cv_sheet.write(f'A{row_index}',roll_num)
				cv_sheet.write(f'B{row_index}','present')
				cv_sheet.write(f'C{row_index}',f'{time.strftime("%H:%M:%S")}')
				#"%s:%s:%s" % (e.hour, e.minute, e.second)

		row_index += 1
		cv2.imshow('im',lst[0])
		if cv2.waitKey(10) & 0xFF==ord('q'):
			break
		#else:
		#pd_detect()##
	cap.release()
	cv2.destroyAllWindows()




"""curnt_date = date.today()
row_index = 2
reg_data = xl.Workbook("attendance.xlsx")
cv_sheet = reg_data.add_worksheet("computer vision")
cv_sheet.write('A1','Roll number')
cv_sheet.write('B1','status')
cv_sheet.write('C1',f'{curnt_date}')
row_index = 2"""

#lod = pd.read_excel(r'regis.xlsx')
#rw_index = 2
#curnt_date = date.today()
#cv_sheet = lod
#size = cv_sheet.shape
#size = size[0]
#print(cv_sheet['Roll number'][0][-1])
#cv_sheet[f'{curnt_date}'] = "Present"
#print(lod)
def exl():
	row_indx = 2
	atten = xl.Workbook("attendance_sheet.xlsx")
	cv_sheet = atten.add_worksheet("computer vision")
	cv_sheet.write('A1','Roll number')
	cv_sheet.write('B1','Status')
	cv_sheet.write('C1','Time')
	pd_detect(cv_sheet,row_indx)
	#atten.drop_duplicates(subset=["Roll number", "status", "Time"], keep="first")
	atten.close()
print("Enter 'r' to start recognition: ")
choice = input()
if choice == 'R' or choice == 'r':
	ret = exl()

#data = pd.read_excel('atten.xlsx')
#data.drop_duplicates(subset=["Roll number", "status", "Time"], keep="first")



