import pymysql
import os 
import pyotp
import cv2
import glob
import shutil
import numpy as np
import time
from datetime import datetime 
import schedule
from PIL import Image
import sys
import numpy as np
import pickle
from multiprocessing import Process
from xml.etree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
import pandas
import face_recognition
from sklearn import svm, neighbors
from joblib import dump, load
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import random
from imutils.video import VideoStream
import imutils
import dlib
import pytesseract
from winreg import *

x = '30/12/2541'
y = '1/1/2541'

z = x.split("/")
print(z[1])

# key = OpenKey(HKEY_CURRENT_USER, 'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders')
# Downloads = QueryValueEx(key, '{374DE290-123F-4565-9164-39C4925E467B}')[0]

# count = len(glob.glob(os.path.join(Downloads,'*jpg')))

# print(Downloads,count)

# ****************************************************************************************************************************

# pytesseract.pytesseract.tesseract_cmd = os.path.join('C:','Program Files','Tesseract-OCR','tesseract.exe')

# img = cv2.imread(os.path.join('F:','data train','test','car2.jpg'),cv2.IMREAD_COLOR)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# gray = cv2.bilateralFilter(gray, 13, 15, 15) 

# edged = cv2.Canny(gray, 30, 200) 
# contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(contours)
# contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
# screenCnt = None

# for c in contours:
    
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.018 * peri, True)
 
#     if len(approx) == 4:
#         screenCnt = approx
#         break

# if screenCnt is None:
#     detected = 0
#     print ("No contour detected")
# else:
#      detected = 1

# if detected == 1:
#     cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

# mask = np.zeros(gray.shape,np.uint8)
# new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
# new_image = cv2.bitwise_and(img,img,mask=mask)

# (x, y) = np.where(mask == 255)
# (topx, topy) = (np.min(x), np.min(y))
# (bottomx, bottomy) = (np.max(x), np.max(y))
# Cropped = gray[topx:bottomx+1, topy:bottomy+1]

# text = pytesseract.image_to_string(Cropped,lang='tha+eng')
# print("programming_fever's License Plate Recognition\n")
# print("Detected license plate Number is:",text)
# img = cv2.resize(img,(500,300))
# Cropped = cv2.resize(Cropped,(400,200))
# cv2.imshow('car',img)
# cv2.imshow('Cropped',Cropped)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ****************************************************************************************************************************

# def openCamera():
#     video_capture = cv2.VideoCapture(0)
#     model = load(os.path.join('F:','data train','model.pkl'))
#     cum = 0

#     while True:
#         _,frame = video_capture.read()

#         face_location = face_recognition.face_locations(frame)
#         face_encoding = face_recognition.face_encodings(frame,face_location)
        
#         names = []
#         for face_encoding_unknows in face_encoding:
            
#             name = model.predict([face_encoding_unknows])

#             names.append(name)
#             # try:
#             #     cv2.imwrite(os.path.join('F:','data train',('training_set%d.jpg' % cum)), frame)
#             # except:
#             #     pass

#         for (top, right, bottom, left), name in zip(face_location, names):
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#             cv2.putText(frame, name[0], (left + 3, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
        
#         cv2.imshow('WEBCAM',frame)

#         cum += 1

#     cv2.destroyAllWindows()

# def checkValue(names):
#     result = {}
#     for name in names:
#         if name in result:
#             result[name] += 1
#         else:
#             result[name] = 1

#     print(result)

# def trainModel(num):
#     encodings = [] 
#     names = [] 
#     name_list = ['non','aof','may']

#     for name in name_list :
#         count = 0
#         file_names = glob.glob(os.path.join('F:','data train',name,'*.jpg'))

#         for file_name in file_names:
#             try:
#                 person_image = face_recognition.load_image_file(file_name)
#                 person_face_encoding = face_recognition.face_encodings(person_image)[0]

#                 encodings.append(person_face_encoding) 
#                 names.append(name)

#                 count +=  1
#             except:
#                 print('ไฟล์เสีย: ',file_name)
#                 pass

#             if count == num:
#                 break
    
#     checkValue(names)

#     model = svm.SVC(probability=True)
#     # model = neighbors.KNeighborsClassifier(n_neighbors = 1)
#     model.fit(encodings,names)
    
#     data = {"encodings": encodings, "names": names}
#     dump(model,os.path.join('F:','data train','model.pkl'))

# def predict():
#     model = load(os.path.join('F:','data train','model.pkl'))

#     # test_images = glob.glob(os.path.join('F:','data train','test','set','*.jpg'))
#     test_images = glob.glob(os.path.join('F:','data train','test','set','*.jpg'))

#     print('\n')
#     x = 0
#     for e in test_images:
#         test_image = face_recognition.load_image_file(e)
#         face_locations_hog = face_recognition.face_locations(test_image,model="hog")
#         for i in zip(face_locations_hog) :
#             test_encoding_hog = face_recognition.face_encodings(test_image,face_locations_hog)[0]
            
#             name = model.predict([test_encoding_hog])
#             probability = model.predict_proba([test_encoding_hog])
#             x = np.ndarray.max(probability) + x
#             print(name, probability,model.classes_,e)

#     print('\n')
#     print(x)
#     print(" %.2f" % float(x/30))

# trainModel(1)
# predict()
# openCamera()

# ****************************************************************************************************************************

# conn = pymysql.connect('localhost','root','','cameraDB')
# cur = conn.cursor()
# sql = 'SELECT linetoken_member.token_id,linetoken_member.token_line FROM linetoken_member INNER JOIN register_member ON linetoken_member.token_id=register_member.token_id WHERE register_member.member_email= %s'
# cur.execute(sql,('ommi406@outlook.com')) 
# data_token = cur.fetchall()
# cur.close()

# line_token = []
# for token in data_token:
#     line_token.append(token[0])
#     line_token.append(token[1])  

# print(line_token)

# def notifyPicture(url):
#     payload = {'message':" ",'imageThumbnail':url,'imageFullsize':url}
#     return _lineNotify(payload)

# def _lineNotify(payload,file=None):
#     import requests
#     url = 'https://notify-api.line.me/api/notify'
#     token = '7NuQEfNJ3j74mnUVEBEturX11KZ7xPkrqL7o7iGpEAu'
#     headers = {'Authorization':'Bearer '+token}
#     return requests.post(url, headers=headers , data = payload, files=file)

# video_capture = cv2.VideoCapture(0)
# c=0
# while True:
#     _,f = video_capture.read()

#     cv2.imshow('cv2',f)

#     if c == 100:
#         notifyPicture('D:\Xampp\htdocs\CAMERA MAN\static\pitpso005@gmail.com\Video\03102020 153155.mp4')
#         c = 0
#     c += 1

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

# ****************************************************************************************************************************

# print(datetime.now().strftime("%d"+"/"+"%m"+"/"+"%Y"),datetime.now().strftime("%X"))

# video_capture = cv2.VideoCapture(0) # ช่องทางแสดงวิดิโอ
# print(video_capture.isOpened())

# fourcc = cv2.VideoWriter_fourcc(*'H264') # ไฟล์วิดิโอ

# # ตัวแปรเริ่มต้น
# face_locations = []
# face_encodings = []

# record_status = 'off' # การบันทึกวิดิโอ
# record_timeStart = 0
# record_timeEnd = 0

# motion_foune = "no" # การตรวจจับการเคลี่อนไหว
# motionDetect_start = 0
# motionDetect_end = 0

# face_found = "no" # การตรวจจับใบหน้า
# faceDetect_start = 0
# faceDetect_end = 0

# fps = 0
# firstFrame = None
# firstFrame_state = 0
# alert_status = 'off'
# path_model = os.path.join('static' ,'pitpso005@gmail.com' ,"Member" ,"Training_Model","model.pkl")

# while True :
#     ret, frame = video_capture.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # เปลี่ยนสีภาพ
#     blur = cv2.GaussianBlur(gray, (21, 21), 0) # เบลอภาพ

#     # กำหนดภาพแรก สำหรับ การตรวจจับการเคลี่อนไหว
#     if firstFrame is None:
#         firstFrame_state += 1 

#         if firstFrame_state == 60:
#             firstFrame = blur
#             firstFrame_state = 0 
#         else:
#             continue

#     names = []
#     alert_status = 'off'

#     if fps == 10:

#         if face_found == 'no':
#             face_locations = face_recognition.face_locations(frame)

#             # ตรวจเจอใบหน้า
#             if face_locations :
#                 face_found = "yes"
#                 face_encodings = face_recognition.face_encodings(frame, face_locations) # ถอดรหัสภาพใบหน้า 128-d
    
#                 for face_encoding_unknows in face_encodings:

#                     # เช็คไฟล์ โมเดล
#                     if os.path.exists(path_model):
#                         model = load(path_model)
#                         confidence = model.predict_proba([face_encoding_unknows])

#                         # เช็คค่าความแม่นยำ ของ การเปรียบเทียวใบหน้า
#                         if (np.ndarray.max(confidence)) >= 0.7 :
#                             name = model.predict([face_encoding_unknows])
#                             print('ตรวจพบใบหน้า '+name[0])

#                         else:
#                             name = ['Unknows']
#                             alert_status = 'on'
#                             print('ตรวจพบใบหน้าที่ไม่รู้จัก')

#                     else:
#                         name = ['Unknows']
#                         print('ตรวจพบใบหน้าที่ไม่รู้จัก')

#                     names.append(name)
                
#                 # วาดกรอบ และ ชื่อ
#                 for (top, right, bottom, left), name in zip(face_locations, names):
#                     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#                     cv2.putText(frame, name[0], (left + 3, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

#                 # cap_frame(frame,image_name,user_email,line_token)

#                 motionDetect_start = 0
#                 motionDetect_end = 0

#                 motion_foune = "no"

#                 faceDetect_start = time.time()
#                 faceDetect_end = time.time()
                
#             else: 
#                 face_found = "no"

#             # ตรวจเจอการเคลี่อนไหว
#             if (face_found == "no") and (motion_foune == 'no'):
#                 frameDelta = cv2.absdiff(firstFrame, blur)
#                 thresh = cv2.dilate(cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
#                 cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#                 for c in cnts:
#                     if cv2.contourArea(c) < 7000:
#                         continue

#                     (x, y, w, h) = cv2.boundingRect(c)
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#                     motionDetect_start = time.time()
#                     motionDetect_end = time.time()
#                     motion_foune = "yes"

#             elif (motion_foune == 'yes') and ((motionDetect_end - motionDetect_start) > 5):
#                 print('ตรวจพบการเคลื่อนไหว')
#                 motionDetect_start = 0
#                 motionDetect_end = 0
#                 motion_foune = "no"

#             elif (motion_foune == 'yes'):
#                 motionDetect_end = time.time()
#                 print('ดีเลการเตลี่อนไหว')
                
#         elif face_found == 'yes' and ((faceDetect_end - faceDetect_start) > 5) :
#             face_found = 'no'

#         elif face_found == 'yes':
#             faceDetect_end = time.time()
#             print('ดีเลใบหน้า')

#         fps = 0
#     else:
#         fps += 1

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     cv2.putText(frame, datetime.now().strftime("%X"+" "+"%x"), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255 ,0), 2, cv2.LINE_8)
#     cv2.imshow('Video', frame)

# video_capture.release()
# cv2.destroyAllWindows()

# ****************************************************************************************************************************

# vs = cv2.VideoCapture(0)

# firstFrame = None
# firstFrame_state = 0

# alert_img = None
# count_img = 0

# motionDetect_start = None
# motionDetect_end = None

# faceDetect_start = None
# faceDetect_end = None
# while True:
#     _,frame = vs.read()

#     face_location = face_recognition.face_locations(frame)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (21, 21), 0)

#     if firstFrame is None:
#         firstFrame_state += 1 

#         if firstFrame_state == 60:
#             firstFrame = gray
#             firstFrame_state = 0 
#         else:
#             continue

#     frameDelta = cv2.absdiff(firstFrame, gray)
#     thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

#     thresh = cv2.dilate(thresh, None, iterations=2)
#     cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for c in cnts:
#         if cv2.contourArea(c) < 10000:
#             continue

#         (x, y, w, h) = cv2.boundingRect(c)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         if alert_img is None:
#             alert_img = frame

#         if (motionDetect_start and motionDetect_end) is None is None:
#             motionDetect_start = time.time()
#             motionDetect_end = time.time()

#     if face_location :
#         face_encord = face_recognition.face_encodings(frame,face_location)

#         for (top, right, bottom, left) in face_location:
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#             motionDetect_start = None
#             motionDetect_end = None
            
#         if (motionDetect_start and motionDetect_end) is not None:
#             motionDetect_end = time.time()

#             if (motionDetect_end-motionDetect_start) > 5:

#                 print('ALERT')
#                 cv2.imwrite('alert_img'+str(count_img)+'.jpg',alert_img)

#                 count_img += 1
#                 alert_img = None
#                 motionDetect_start = None
#                 motionDetect_end = None

#     cv2.imshow("Security Feed", frame)
#     cv2.imshow("Thresh", thresh)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cv2.destroyAllWindows()

# ****************************************************************************************************************************

# conn = pymysql.connect('localhost','root','','cameraDB')
# with conn :
#     cur = conn.cursor()
#     sql = 'SELECT videos_name,videos_date,videos_timeStart,videos_timeEnd FROM videos_member WHERE member_email = %s'
#     cur.execute(sql,('pitpso005@gmail.com')) 
#     rows = cur.fetchall()
#     cur.close()

# name_list = [] # สร้าง list เก็บข้อมูล ชื่อวิดิโอ ที่ SELECT ออกมา# สร้าง list เก็บข้อมูล วันวิดิโอ ที่ SELECT ออกมา
# for row in rows:
#     name_list.append(row[0]) # row[0] = ข้อมูล ชื่อ

# name_list = list(dict.fromkeys(name_list)) 
# print(name_list)  
 
# file_names = glob.glob1(os.path.join('static','pitpso005@gmail.com','Video'),'*') 
# for file_name in file_names:
#     if file_name in name_list:
#         pass
#     else:
#         print(file_name)   

# ****************************************************************************************************************************

# num = 90
# while num:
#     mins, secs = divmod(num, 60)
#     timeformat = '{:02d}:{:02d}'.format(mins, secs)
#     print(timeformat, end='\r')
#     time.sleep(1)
#     num -= 1
# print('Goodbye!\n\n\n\n\n')


# ****************************************************************************************************************************

# path_model = os.path.join('static' ,'pitpso005@gmail.com' ,"Member" ,"Training_Model","model.pkl")
# file_names = os.path.join('F:','data train','test','ing.jpg')
# known_image = face_recognition.load_image_file(file_names)
# face_location = face_recognition.face_locations(known_image,model="hog")
# face_encode = face_recognition.face_encodings(known_image,face_location)
# print(face_location,len(face_location))
# print(face_encode,len(face_encode))
# model = load(path_model)
# for i in face_encode:
#     name = model.predict([i])
#     confidence = model.predict_proba([i])
#     if (np.ndarray.max(confidence)) >= 0.7 :
#         print(name,confidence)
#     else:
#         name = 'unknows'
#         print(name,confidence)

# ****************************************************************************************************************************

# video_capture = cv2.VideoCapture(0)
# state = '0'
# while True :
#     name = datetime.now().strftime("%d%m%Y %H%M%S") + '.mp4'

#     ret, frame = video_capture.read()

#     if ((time_end - time_start) > 10 ) or state == '1':
#         if state == '0':
#             fourcc = cv2.VideoWriter_fourcc(*'H246')
#             out = cv2.VideoWriter(name,fourcc, 20.0, (640,480))
#             state = "1"
#             time_start = time.time()
#             time_end = time.time()

#             print('บันทึกวิดิโอ')

#         elif ((time_end - time_start) > 5 ) and state == '1':
#             out.release()
#             state = "0"
#             time_start = time.time()
#             time_end = time.time()
#             print('เลิกบันทึกวิดิโอ')
#             time.sleep(3)

#         elif state == '1':
#             out.write(frame)
#             print('โหลดวิดิโอ')

#     print((time_end - time_start))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     cv2.imshow('Video', frame)
    
# video_capture.release()
# cv2.destroyAllWindows()


# ****************************************************************************************************************************

# video_capture = cv2.VideoCapture(0)
# i = 0

# while i <= 1000 :
#     ret, frame = video_capture.read()
#     i += 1

#     try:
#         cv2.imwrite(os.path.join('F:','data train',('training_set%d.jpg' % i)), frame)
#     except:
#         pass

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     cv2.imshow('Video', frame)
#     time.sleep(1)
# video_capture.release()
# cv2.destroyAllWindows()

# ****************************************************************************************************************************

# x = [16,24,33,40,56,15,38]
# print(max(x))
# print(x.index(max(x)))

# ****************************************************************************************************************************

# def face_detect(test):
#     video_capture = cv2.VideoCapture(0)
#     while True:
#         faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_dataset.xml')
#         detector = dlib.get_frontal_face_detector()

#         ref,frame = video_capture.read()

#         image = face_recognition.load_image_file(test)

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         face_opencv = faceCascade.detectMultiScale(
#             gray,
#             scaleFactor=1.1,
#             minNeighbors=5,
#             minSize=(30, 30),
#             flags=cv2.CASCADE_SCALE_IMAGE
#         )
#         face_dlib = face_recognition.face_locations(frame)

#         dets = detector(frame, 1)
#         for d in dets:
#             xy = d.left(), d.top()
#             wh = d.right(), d.bottom()
#             cv2.rectangle(frame,xy,wh,(255,0,255),2)

#         for (top, right, bottom, left) in face_dlib:
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#         for (x, y, w, h) in face_opencv:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         print('openCV: ',face_opencv)
#         print('Dlib: ',face_dlib)
#         print('dets: ',dets)

#         cv2.imshow('image',frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cv2.destroyAllWindows()

# def trainModel():
#     encodings = [] 
#     names = [] 
#     name_list = ['aof','non','may','nack','cheg']

#     for name in name_list :
#         # print(name)
#         count = 0
#         file_names = glob.glob(os.path.join('F:','data train',name,'*.jpg'))

#         for file_name in file_names:
#             try:
#                 person_image = face_recognition.load_image_file(file_name)
#                 person_face_encoding = face_recognition.face_encodings(person_image)[0]

#                 encodings.append(person_face_encoding) 
#                 names.append(name)
#             except:
#                 count +=  1
#                 print('ไฟล์เสีย: ',file_name)
#                 pass
#         print(count)

#     # Create and train the SVC classifier 
#     model_knn = neighbors.KNeighborsClassifier(n_neighbors = 1)
#     model_knn.fit(encodings,names) 

#     model_svm = svm.SVC(probability=True)
#     model_svm.fit(encodings,names) 
    
#     data = {"encodings": encodings, "names": names}
#     # create and calling model flie
#     dump(model_svm,os.path.join('F:','data train','model_svm.pkl'))
#     dump(model_knn,os.path.join('F:','data train','model_knn.pkl'))

# def face_recognize(test): 
#     model_svm = load(os.path.join('F:','data train','model_svm.pkl'))
#     model_knn = load(os.path.join('F:','data train','model_knn.pkl'))

#     # Load the test image with unknown faces into a numpy array 
#     test_image = face_recognition.load_image_file(test)
  
#     # Find all the faces in the test image using the default HOG-based model 
#     face_locations_hog = face_recognition.face_locations(test_image,model="hog")
#     # face_locations_cnn = face_recognition.face_locations(test_image,model="cnn")

#     no = len(face_locations_hog) 
#     print("Number of faces detected: ", no) 
  
#     # Predict all the faces in the test image using the trained classifier
#     print("Found:") 
#     for i in range(no): 
#         test_image_enc_hog = face_recognition.face_encodings(test_image,face_locations_hog)[i]
#         # test_image_enc_cnn = face_recognition.face_encodings(test_image,face_locations_cnn)[i]
        
#         name_svm = model_svm.predict([test_image_enc_hog])
#         confidence_svm = model_svm.predict_proba([test_image_enc_hog])

#         name_knn = model_knn.predict([test_image_enc_hog])
#         confidence_knn = model_knn.predict_proba([test_image_enc_hog])

#         if (np.ndarray.max(confidence_svm)) >= 0.7 :
#             print('KNN: ',name_knn,confidence_knn)
#             print('SVM: ',name_svm,confidence_svm)
#         else:
#             print('KNN: ',name_knn,confidence_knn)
#             print('SVM: ','Unknows',confidence_svm)
#         print('.......................................................................')
#     # for (top, right, bottom, left) in face_locations_hog:
#     #     cv2.rectangle(test_image, (left, top), (right, bottom), (255, 0, 0), 2)
    
#     # cv2.imwrite('result.jpg',cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
#     # cv2.waitKey(0)
#     # cv2.destroyWindow()

# def main():
#     test_image = glob.glob(os.path.join('F:','data train','test','must.jpg'))
#     # trainModel()
#     # face_recognize(test_image[0]) 
#     # face_detect(test_image[0])
# if __name__=="__main__": 
#     main() 

# ****************************************************************************************************************************

# root = ElementTree.parse('data.xml').getroot()
# for data in root:
#     print(data.attrib)

# def create_dataset(name,datas):

#     title = Element('data') # ประกาศ Element

#     member = SubElement(title, 'member', name=name) #เพิ่ม Element ย่อย
    
#     for data in datas :
#         detail = SubElement(member, 'member', name=name)

#     output_file = open('data.xml', 'w') #สร้างไฟล์ data.xml

#     output_file.write(ElementTree.tostring(title).decode('utf-8')) #เขียนไฟล์ xml 

#     output_file.close()

# ****************************************************************************************************************************

# x = open('data_set_non.txt','r').read()
# # print(x.split(',')[0])

# file_names = glob.glob(os.path.join(os.getcwd(),'static','pitpso005@gmail.com','Member',"non",'Data_set','*.jpg'))
# person_image = face_recognition.load_image_file(file_names[0])
# person_face_encoding = face_recognition.face_encodings(person_image)[0]

# f = open('data_set_non.txt', 'r').read()

# matches = face_recognition.compare_faces([f.split(',')[0]], person_face_encoding,0.5)

# print(matches)
# f.close()

# ****************************************************************************************************************************

# name_list = ['non','aof','ing']

# for name in name_list :
#     print(name)
#     file_names = glob.glob(os.path.join(os.getcwd(),'static','pitpso005@gmail.com','Member',name,'Data_set','*.jpg'))

#     files = open(('data_set_'+name+'.txt'), 'w')

#     for file_name in file_names:
#         try:
#             person_image = face_recognition.load_image_file(file_name)
#             person_face_encoding = face_recognition.face_encodings(person_image)[0]

#             files.write(' %s ,' %(person_face_encoding))
#         except:
#             print('ไฟล์เสีย: ',file_name)

#     files.close()
# print('เสร็จสิ้น')

# ****************************************************************************************************************************

# f = open('data_set_non.txt', 'r')
# for i in f :
#     print(i)
#     time.sleep(2)
# f.close()

# ****************************************************************************************************************************

# video_capture = cv2.VideoCapture(0)

# while True:
#     ret, frame = video_capture.read()

#     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

#     rgb_small_frame = small_frame[:, :, ::-1]

#     face_locations_unknows = face_recognition.face_locations(rgb_small_frame)
#     face_encodings_unknows = face_recognition.face_encodings(rgb_small_frame, face_locations_unknows)

#     names = []
#     for face_encoding_unknows in face_encodings_unknows:
#         model = load('model.pkl')
#         confidence = model.predict_proba([face_encoding_unknows])

#         if (np.ndarray.max(confidence)) >= 0.6 :
#             name = model.predict([face_encoding_unknows])
    
#         else:
#             name = ['Unknows']

#         names.append(name)

#     for (top, right, bottom, left), name in zip(face_locations_unknows, names):
#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4

#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         cv2.putText(frame, name[0], (left + 3, bottom - 3), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     cv2.imshow('Video', frame)
    
# video_capture.release()
# cv2.destroyAllWindows()

# ****************************************************************************************************************************

# import multiprocessing

# def job():
#     print('unknows')
#     time.sleep(5)


# if __name__ == "__main__":

# ****************************************************************************************************************************

# # เปิดการใช้ webcam
# video_capture = cv2.VideoCapture(0)

# # โหลดภาพ Peen.jpg และให้ระบบจดจำใบหน้า
# files = glob.glob(os.path.join(os.getcwd(),'static','pitpso005@gmail.com','Member','non','*jpg'))

# person1_image = face_recognition.load_image_file(files[0])
# person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

# # สร้าง arrays ของคนที่จดจำและกำหนดชื่อ ตามลำดับ
# known_face_encodings = [ person1_face_encoding ]

# known_face_names = ["non"]

# # ตัวแปรเริ่มต้น
# face_locations = []
# face_encodings = []
# face_names = []
# process_this_frame = True

# while True:
#     # ดึงเฟรมภาพมาจากวีดีโอ
#     ret, frame = video_capture.read()

#     # ย่อขนาดเฟรมเหลือ 1/4 ทำให้ face recognition ทำงานได้เร็วขึ้น
#     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

#     # แปลงสีภาพจาก BGR (ถูกใช้ใน OpenCV) เป็นสีแบบ RGB (ถูกใช้ใน face_recognition)
#     rgb_small_frame = small_frame[:, :, ::-1]

#     # ประมวลผลเฟรมเว้นเฟรมเพื่อประหยัดเวลา
#     if process_this_frame:
#         # ค้นหาใบหน้าที่มีทั้งหมดในภาพ จากนั้นทำการ encodings ใบหน้าเพื่อจะนำไปใช้เปรียบเทียบต่อ
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#         face_names = []
#         for face_encoding in face_encodings:
#             # ทำการเปรียบเทียบใบหน้าที่อยู่ในวีดีโอกับใบหน้าที่รู้จักในระบบ
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#             name = "Unknown"

#             # ถ้า encoding แล้วใบหน้าตรงกันก็จะแสดงข้อมูล
#             if True in matches:
#                 first_match_index = matches.index(True)
#                 name = known_face_names[first_match_index]

#             else: 
#                 name = "Unknown"         

#             face_names.append(name)

#     process_this_frame = not process_this_frame

#     # แสดงผลลัพธ์
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         # ขยายเฟรมที่ลดลงเหลือ 1/4 ให้กลับไปอยู่ในขนาดเดิม 
#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4

#         # วาดกล่องรอบใบหน้า
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#         # เขียนตัวหนังสือที่แสดงชื่อลงที่กรอบ
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

#     cv2.putText(frame,datetime.now().strftime("%X"+" "+"%x"),(10, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255 ,0),2,cv2.LINE_8)
        
#     # แสดงรูปภาพผลลัพธ์
#     cv2.imshow('Video', frame)

#     # กด 'q' เพื่อปิด!
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
            
# video_capture.release()
# cv2.destroyAllWindows()


# ****************************************************************************************************************************

# def render_frame(user_email,line_token,alert_delay,record_delay):
#     faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_dataset.xml')

#     while video_capture.isOpened():
#         img_name = time_now() # เวลาที่ใช้ในการ จัดเก็บข้อมูล
#         show_time = datetime.now().strftime("%X"+" "+"%x") # เวลาที่ใช้ในการ ส่งข้อมูล

#         ret, frame = video_capture.read()

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

#         faces = faceCascade.detectMultiScale(
#             gray,
#             scaleFactor=1.1,
#             minNeighbors=5,
#             minSize=(30, 30),
#             flags=cv2.CASCADE_SCALE_IMAGE
#         )

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # เขียนกรอบรอบใบหน้า

#             # save_video(img_name,record_delay,user_email) # บันทึกวิดิโอลงใน โฟลเดอร์ 

#             # try:
#             #     cap_frame(frame,img_name,user_email) # บันทึกรูปลงใน โฟลเดอร์ และ database

#             #     files = sorted(glob.glob(os.path.join(os.getcwd(),'static',user_email,'Alert','*jpg')) ,key = os.path.getmtime) # แสดงชื่อภาพล่าสุดในโฟล์เดอร์

#             #     send_image(show_time ,files[-1] ,line_token) # ส่ง รูปและเวลา ไปทาง Line

#             #     time.sleep(alert_delay)
#             # except:
#             #     pass

#         cv2.putText(frame,show_time,(10, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255 ,0),2,cv2.LINE_8) # แสดงเวลาที่ รูป

#         yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg',frame)[1].tobytes() + b'\r\n')

# ****************************************************************************************************************************

# start = time.time()
# end = time.time()

# while (end-start) < 10:
#     end =  time.time()
#     print(end - start)

# def get_list(x,y):
#     array = []
#     array.append(x)
#     array.append(y)

#     return array

# print(get_list(10,15)[1])

# def time_now(): # แสดงเวลาปัจจุบัน
#     now = datetime.now()
#     date_time = now.strftime("%d%m%Y %H%M%S")
#     return date_time

# print( len(time_now()) )

# files = sorted(glob.glob(os.path.join(os.getcwd(),'static','pitpso005@gmail.com','Alert','*jpg')),key=os.path.getmtime)
# print(files[-1])

# app = Flask(__name__)

# line_bot_api = LineBotApi('1J5q5f/BUVFTU6To3KTGh49nC9pCPtLxmS49y1UuFKm/IGoCB4SPfIlQSeCdM7Z17/n3zqE34UGya7UU45H0ooTqcVfytOJLsGvUO3mJbj6Qa9nbSaXisipHGmAha90mLtMW5dNlObgeodnnuGF3HgdB04t89/1O/w1cDnyilFU=')
# handler = WebhookHandler('75d10d7278b6dbe05ecececaed6c581d')


# @app.route("/callback", methods=['POST'])
# def callback():
#     # get X-Line-Signature header value
#     signature = request.headers['X-Line-Signature']

#     # get request body as text
#     body = request.get_data(as_text=True)
#     app.logger.info("Request body: " + body)

#     # handle webhook body
#     try:
#         handler.handle(body, signature)
#     except InvalidSignatureError:
#         print("Invalid signature. Please check your channel access token/channel secret.")
#         abort(400)

#     return 'OK'


# @handler.add(MessageEvent, message=TextMessage)
# def handle_message(event):
#     line_bot_api.reply_message(
#         event.reply_token,
#         TextSendMessage(text=event.message.text))


# if __name__ == "__main__":
#     app.run()

# *******************************************************************************************************************************

# def generate(image_path):
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_dataset.xml')
#     img = cv2.imread(image_path)
#     if (img is None):
#             print("Can't open image file")
#             return 0

#     faces = face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
#     if (faces is None):
#         print('Failed to detect face')
#         return 0

#     facecnt = len(faces)

#     if facecnt == 0:
#         print('not found face')
#         return 0

#     print("Detected faces: %d" % facecnt)
#     i = 0
#     height, width = img.shape[:2]

#     for (x, y, w, h) in faces:
#         r = max(w, h) / 2
#         centerx = x + w / 2
#         centery = y + h / 2
#         nx = int(centerx - r)
#         ny = int(centery - r)
#         nr = int(r * 2)

#         faceimg = img[ny:ny+nr, nx:nx+nr]
#         lastimg = cv2.resize(faceimg, (50,50))
#         i += 1
#         cv2.imwrite("training_set%d.jpg" % i, lastimg)

# generate(os.path.join('F:','data train','test','must.jpg'))

# *******************************************************************************************************************************

# mylist = ["a", "b", "a", "c", "c","b","c"]
# mylist = list(dict.fromkeys(mylist))
# print(mylist)

# files = sorted(glob.glob(os.path.join('static','pitpso005@gmail.com','Alert','*jpg')))
# # files = sorted(glob.glob1(os.path.join('static','pitpso005@gmail.com','Alert'),'*jpg'))
# for i in files :
#     print(i.split('\\')[3])


# date = datetime.now().strftime("%Y"+"-"+"%m"+"-"+"%d")
# time = datetime.now().strftime("%X")
# print(len(time))
# print(time)
# if x == "2020-07-16" :
#     print('y')

# schedule.every(2).seconds.do(lambda: non(sum))
# sum = 0
# while 1:
#     schedule.run_pending()
#     sum += 1
# # print(f)

# *******************************************************************************************************************************

# def time_now():
#     now = datetime.now()
#     date_time = now.strftime("%m%d%Y %H%M%S")
#     return date_time

# while 1:
#     print(time_now())


# *******************************************************************************************************************************

# t0 = time.time()
# # print("t0: ",t0)
# while True:
#    t1 = time.time()
# #    print("t1: ",t1)
#    num_seconds = t1 - t0
#    time.sleep(1)
#    if num_seconds > 5:
#        print("...")
#        t0 = time.time()

# *******************************************************************************************************************************
# def test():
#     cap = cv2.VideoCapture(0)

#     fourcc = cv2.VideoWriter_fourcc(*'H264')
#     out = cv2.VideoWriter("static/pitpso005@gmail.com/Video/test.mp4",fourcc, 30.0, (640,480))

#     while True:
#         ret, frame = cap.read()

#         out.write(frame)

#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     out.release()
#     cap.release()
#     cv2.destroyAllWindows()

# def webcam():
#     cap = cv2.VideoCapture(0)
    
#     time_start = time.time()

#     while(cap.isOpened()):

#         time_now = time.time()
#         time_count = time_now - time_start

#         ret, frame = cap.read()

#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         if time_count > 10:
#             test()
#     cap.release()
#     cv2.destroyAllWindows()

# webcam()
# test()
# *******************************************************************************************************************************

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('non.avi',fourcc, 30.0, (640,480))

# # เปิดการใช้ webcam
# video_capture = cv2.VideoCapture(0)

# # โหลดภาพ Peen.jpg และให้ระบบจดจำใบหน้า
# person1_image = face_recognition.load_image_file("test/non.jpg")
# person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

# # สร้าง arrays ของคนที่จดจำและกำหนดชื่อ ตามลำดับ
# known_face_encodings = [
#     person1_face_encoding
# ]

# known_face_names = [
#     "non"
# ]

# # ตัวแปรเริ่มต้น
# face_locations = []
# face_encodings = []
# face_names = []
# process_this_frame = True

# while True:
#     # ดึงเฟรมภาพมาจากวีดีโอ
#     ret, frame = video_capture.read()

#     # ย่อขนาดเฟรมเหลือ 1/4 ทำให้ face recognition ทำงานได้เร็วขึ้น
#     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

#     # แปลงสีภาพจาก BGR (ถูกใช้ใน OpenCV) เป็นสีแบบ RGB (ถูกใช้ใน face_recognition)
#     rgb_small_frame = small_frame[:, :, ::-1]

#     # ประมวลผลเฟรมเว้นเฟรมเพื่อประหยัดเวลา
#     if process_this_frame:
#         # ค้นหาใบหน้าที่มีทั้งหมดในภาพ จากนั้นทำการ encodings ใบหน้าเพื่อจะนำไปใช้เปรียบเทียบต่อ
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#         face_names = []
#         for face_encoding in face_encodings:
#             # ทำการเปรียบเทียบใบหน้าที่อยู่ในวีดีโอกับใบหน้าที่รู้จักในระบบ
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#             name = "Unknown"

#             # ถ้า encoding แล้วใบหน้าตรงกันก็จะแสดงข้อมูล
#             if True in matches:
#                 first_match_index = matches.index(True)
#                 name = known_face_names[first_match_index]

#             face_names.append(name)

#     process_this_frame = not process_this_frame

#     # แสดงผลลัพธ์
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         # ขยายเฟรมที่ลดลงเหลือ 1/4 ให้กลับไปอยู่ในขนาดเดิม 
#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4

#         # วาดกล่องรอบใบหน้า
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#         # เขียนตัวหนังสือที่แสดงชื่อลงที่กรอบ
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

#     # แสดงรูปภาพผลลัพธ์
#     out.write(frame)
#     cv2.imshow('Video', frame)

#     # กด 'q' เพื่อปิด!
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# out.release()
# video_capture.release()
# cv2.destroyAllWindows()

# *******************************************************************************************************************************

# now = datetime.now()
# current_time1 = now.strftime("%H:%M:%S")
# print("Current Time =", current_time1)

# t = time.localtime()
# current_time2 = time.strftime("%H:%M:%S", t)
# print(current_time2)

# for i in range(10,0,-1):
#     time.sleep(1)
#     print (i)

# *******************************************************************************************************************************

# person1 = face_recognition.load_image_file(os.path.join('F:','data train','test','non.jpg'))
# person2 = face_recognition.load_image_file(os.path.join('F:','data train',"non",'training_set3.jpg'))

# face_locations1 = face_recognition.face_encodings(person1)[0]
# face_locations2 = face_recognition.face_encodings(person2)[0]

# result = face_recognition.compare_faces([face_locations1],face_locations2,tolerance=0.6)
# ac = face_recognition.face_locations(person1,model='cnn')
# print(result,ac)

# *******************************************************************************************************************************

# print(os.getcwd())
# os. chdir("D:\Xampp\htdocs\camera man\static")
# print(os.getcwd())

# path = os.path.join("d:\Xampp\htdocs\camera man\static","non") 
# try:
#     os.mkdir(path)
#     print("สร้าง")
# except:
#     os.rmdir(path)
#     print("ลบ")

# *******************************************************************************************************************************

# files = glob.glob(os.path.join(os.getcwd(),'static','pitpso005@gmail.com','Alert','*jpg'))
# print(files[-1])

# for f1 in files:
# #     print(f1)

# name = 'non'
# path = os.path.join("d:\Xampp\htdocs\camera man\static"+"\\"+"pitpso005@gmail.com",name) 
# try:
#     os.mkdir(path)
#     # shutil.copy("F:\Fate-Grand-Order-Gudako-gacha.jpg","D:\Xampp\htdocs\camera man\static\pitpso005@gmail.com\non") 
#     print('สร้าง')
# except:
#     os.rmdir(path)
#     print('ลบ')

# src_dir = "F:\\"
# dst_dir = "D:\Xampp\htdocs\camera man\static\pitpso005@gmail.com\\non"
# for jpgfile in sorted(glob.iglob(os.path.join(src_dir, "*.jpg"))):
#     shutil.copy(jpgfile, dst_dir)
#     print(jpgfile)

# name="non"
# folder = "pitpso005@gmail.com"
# path = "D:\Xampp\htdocs\camera man\static\pitpso005@gmail.com"

# shutil.copy("‪F:\_MG_0682.JPG" , os.getcwd())

# image = "F:\\"+"aof.jpg"
# image.save(os.path.join(os.getcwd(), image.filename))

# *******************************************************************************************************************************

# base32secret = pyotp.random_base32() 
# gen_otp = pyotp.TOTP(base32secret)
# print(gen_otp.now())
# time.sleep(10)
# gen_otp = pyotp.TOTP(base32secret)
# print(gen_otp.now())