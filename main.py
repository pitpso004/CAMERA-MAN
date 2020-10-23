from flask import Flask, flash, render_template as render, request, redirect, url_for, session, Response, json
from datetime import datetime 
from send_mail import send_forgetpassword , send_otp
from line import send_line , send_image
from werkzeug.utils import secure_filename
import pymysql # เชื่อมต่อ Database
import os ,shutil ,glob # จัดการไฟล์ต่างๆ
import cv2
import face_recognition
import random
import time
import numpy as np
from sklearn import svm, neighbors
from joblib import dump, load 

def create_model(user_email):
    encodings = []
    names = []

    cur = conn.cursor()
    sql = 'SELECT image_folder FROM images_member WHERE member_email = %s'
    cur.execute(sql,(user_email)) 
    datas = cur.fetchall()

    name_list = []
    for data in datas :
        name_list.append(data[0])

    name_list = list(dict.fromkeys(name_list))

    for name in name_list :
        file_names = glob.glob(os.path.join(os.getcwd(),'static',session['email'],'Member',name,'*.jpg'))

        for file_name in file_names:
            try:
                person_image = face_recognition.load_image_file(file_name)
                person_face_encoding = face_recognition.face_encodings(person_image)[0]

                encodings.append(person_face_encoding) 
                names.append(name)
            except:
                os.remove(file_name)
                print('ไฟล์เสีย: ',file_name)
                pass

    model = svm.SVC(probability=True)
    model.fit(encodings, names) 
    
    dump(model,os.path.join('static' ,user_email ,"Member" ,"Training_Model","model.pkl"))
    cur.close()

    print('สร้างโมเดล สำเร็จ')

def get_linetoken(user_email): # ขอ line token ข้อผู้ใช้
    cur = conn.cursor()
    sql = 'SELECT linetoken_member.token_id,linetoken_member.token_line FROM linetoken_member INNER JOIN register_member ON linetoken_member.token_id=register_member.token_id WHERE register_member.member_email= %s'
    cur.execute(sql,(user_email)) 
    data_token = cur.fetchall()
    cur.close()

    line_token = []
    for token in data_token:
        line_token.append(token[0]) # line_token[0] = token id 
        line_token.append(token[1]) # line_token[1] = line_token

    return line_token


def get_delay(user_email):
    cur = conn.cursor()
    sql = 'SELECT delay_alert,delay_record FROM register_member WHERE member_email = %s'
    cur.execute(sql,(user_email)) 
    rows = cur.fetchall()
    cur.close()

    delay = []
    for row in rows:
        delay.append(row[0]) # delay[0] = alert_delay
        delay.append(row[1]) # delay[1] = record_delay

    return delay

def time_now(): # แสดงเวลาปัจจุบัน
    now = datetime.now()
    date_time = now.strftime("%d%m%Y %H%M%S")
    return date_time
    
def create_folde(user_email):
    os.mkdir(os.path.join(os.getcwd(),'static' ,user_email)) # สร้าง folder ของผู้ใช้งาน
    os.mkdir(os.path.join(os.getcwd(),'static',user_email ,"Alert")) # สร้าง folder เก็บประวัติการแจ้งเตือน
    os.mkdir(os.path.join(os.getcwd(),'static',user_email ,"Video")) # สร้าง folder เก็บวิดิโอที่บันถึก
    os.mkdir(os.path.join(os.getcwd(),'static',user_email ,"Member")) # สร้าง folder ของสมาชิก
    os.mkdir(os.path.join(os.getcwd(),'static',user_email ,"Member" ,"Training_Model")) # สร้าง folder Training_Model ในโฟลเดอร์ สมาชิก

conn = pymysql.connect('localhost','root','','cameraDB')

app = Flask(__name__, static_folder='static')
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

video_capture = cv2.VideoCapture(0) # เปิดการใช้ webcam

fourcc = cv2.VideoWriter_fourcc(*'H264') # ไฟล์วิดิโอ

@app.route("/") # หน้า main
def main():
    return render('main.html')

@app.route("/login") # หน้าฟอร์ม Login
def login():
    alert=""
    return render('login.html',alert=alert)

@app.route("/logout") # หน้า logout
def logout():
    cur = conn.cursor()
    sql = 'UPDATE register_member SET member_state = "0" WHERE member_email = %s'
    cur.execute(sql,(session['email']))
    conn.commit()

    session.clear()
    return redirect(url_for('main'))
                    
@app.route("/register") # หน้าฟอร์ม สมัครสมาชิก
def register():
    return render('register.html')

@app.route("/confirmRegister") # หน้าฟอร์ม otp ยืนยัน อีเมล จากการ สมัครสมาชิก
def confirmRegister():
    return render('confirm_register.html')

@app.route("/confirmForgetpassword") # หน้าฟอร์ม otp ยืนยัน อีเมล จากการ ลืมรหัสผ่าน
def confirmForgetpassword():
    return render('confirm_forgetpassword.html')

@app.route("/forgetPassword") # หน้าฟอร์ม ลืม รหัสผ่าน
def forget():
    return render('forget_password.html')

@app.route("/checkLogin",methods = ['post']) # ตรวจสอบ การลงทะเบียนเข้าใช้งาน
def checkLogin():
    if request.method == "POST":
        user_email = request.form['email']
        user_password = request.form['password']
        count = "0"


        cur = conn.cursor()
        cur.execute('SELECT * FROM register_member') 
        rows = cur.fetchall()

        for row in rows : # ค้นหาข้อมูลใน database
            if user_email == row[2] and user_password == row[3]: # เปรียบเทียบข้อมูลที่กรอกมา กับ ค่าที่ค้นหา

                if row[4] == '0':  # เช็คค่าข้อมูลใน data base
                    cur.execute('UPDATE register_member SET member_state = %s WHERE member_email = %s',('1',user_email))
                    conn.commit()
                    cur.close()

                    session['name'] = row[1] # email
                    session['email'] = row[2] # password
                    
                    path_model = os.path.join('static',user_email,"Member","Training_Model","model.pkl")
                    if os.path.exists(path_model):
                        return redirect(url_for('home'))
                    else:
                        return redirect(url_for('member'))
                else:
                    alert = "ชื่อผู้ใช้นี้กำลังใช้งานอยู่"
                    return redirect(url_for('login',alert=alert))
            else:
                count = "0"

        if count == "0":
            alert = "อีเมล หรือ รหัสผ่านไม่ถูกต้อง"
            return redirect(url_for('login',alert=alert))

@app.route("/checkRegister",methods = ['post']) # ตรวจสอบ การสมัครสมาชิก
def checkRegister():
    if request.method == "POST":
        session['name'] = request.form['name']
        session['email'] = request.form['email']
        session['password'] = request.form['password']

        cur = conn.cursor()
        cur.execute('SELECT member_email FROM register_member')
        rows = cur.fetchall()
        cur.close()

        if rows != 0 :
            for row in rows : # ค้นหา และ เปรียบเทียบข้อมูล
                if (session['email'] == row[0]):
                    count = '0'
                    break
                else:
                    count = '1'
        else:
            count = '1'

        if count == '1' :
            session['gen_otp'] = str(random.randint(100000, 999999)) # สร้างรหัส OTP
            print(session['gen_otp'])

            send_otp(session['email'],session['gen_otp'],'confirmRegister') # ส่งรหัส OTP ไปยังอีเมลของผู้สมัครสมาชิก
                
            alert = "ระบบได้ทำการส่งรหัส OTP ไปยังอีเมลดังกล่าว"
            return redirect(url_for('confirmRegister',confirm=alert))

        else:
            alert = "อีเมล นี้ถูกใช้งานแล้ว"
            return redirect(url_for('register',alert=alert))

@app.route("/sendOtp_confirmRegister",methods=['post','get']) # ส่งรหัส otp ใหม่ confirmRegister
def sendOtp_confirmRegister():
    try:
        if session['email']:

            session['gen_otp'] = str(random.randint(100000, 999999)) # สร้างรหัส OTP
            send_otp(session['email'],session['gen_otp'],'confirmRegister') # ส่งรหัส OTP ไปยังอีเมลของผู้สมัครสมาชิก
            print(session['gen_otp'])

            alert = "ระบบได้ทำการส่งรหัส OTP ไปยังอีเมลดังกล่าว"
            return redirect(url_for('confirmRegister',confirm=alert))
    except:
        alert = "ไม่สามารถส่งรหัส OTP ได้"
        return redirect(url_for('confirmRegister',alert=alert))

@app.route("/checkOtp_register",methods = ['post']) # ตรวจสอบ รหัส OTP จาก การสมัครสมาชิก
def checkOtp_register():
    if request.method == "POST":
        user_otp = request.form['otp'] # OTP ที่ผู้ใช้กรอกเข้ามา

        gen_otp = session['gen_otp'] # OTP ที่ระบบสร้าง
        user_name = session['name']
        user_email = session['email']
        user_password = session['password']

        if gen_otp == user_otp : # เปรียนเทียบ otp ที่ผู้สมัครสมาชิกกรอกเข้ามา กับ otp ที่ระบบสร้างขึ้น
            cur = conn.cursor()
            sql = "INSERT INTO register_member (member_name,member_email,member_password,member_state,token_id,delay_alert,delay_record) VALUES (%s,%s,%s,%s,%s,%s,%s)" # เพิ่มค่าในตาราง register_member
            cur.execute(sql,(user_name,user_email,user_password,'0',0,1,60))
            conn.commit()
            cur.close()

            try:
                create_folde(user_email)
            except:
                shutil.rmtree(os.path.join(os.getcwd(),"static",user_email))
                create_folde(user_email)
                
            session.clear()
            alert = "สมัครสมาชิดเรียบร้อย"
            return redirect(url_for('register',confirm=alert))
        else:
            alert = "รหัส OTP ไม่ถูกต้อง"
            return redirect(url_for('confirmRegister',alert=alert))        

@app.route("/selectPassword",methods = ['post']) # ตรวจสอบ อีเมล ที่ต้องการขอ รหัสผ่าน
def selectPassword():
    if request.method == "POST": # รับ อีเมล จากผู้ใช้
        user_email = request.form['email']
        count = '0'

    cur = conn.cursor()
    cur.execute('SELECT member_email,member_password FROM register_member')
    rows = cur.fetchall()
    cur.close()

    if len(rows) != 0 :
        for row in rows :
            if (user_email == row[0]):
                session['email'] = user_email
                session['password'] = row[1]

                session['gen_otp'] = str(random.randint(100000, 999999)) # สร้างรหัส OTP
                send_otp(session['email'],session['gen_otp'],'confirmForgetpassword') # ส่งรหัส OTP ไปยังอีเมล
                print(session['gen_otp'])

                count = '1'
                break
            else:
                count = '0'

    if count == '1' :
        alert = "ระบบได้ทำการส่งรหัส OTP ไปยังอีเมลดังกล่าว"
        return redirect(url_for('confirmForgetpassword',confirm=alert))
    else:
        alert = "ไม่พบ อีเมล ดังกล่าว"
        return redirect(url_for('forget',alert=alert))

@app.route("/sendOtp_confirmForgetpassword",methods=['post','get']) # ส่งรหัส otp ใหม่ confirmForgetpassword
def sendOtp_confirmForgetpassword():
    try:
        if session['email']:
            session['gen_otp'] = str(random.randint(100000, 999999)) # สร้างรหัส OTP
            send_otp(session['email'],session['gen_otp'],'confirmForgetpassword') # ส่งรหัส OTP ไปยังอีเมลของผู้สมัครสมาชิก
            print(session['gen_otp'])

            alert = "ระบบได้ทำการส่งรหัส OTP ไปยังอีเมลดังกล่าว"
            return redirect(url_for('confirmForgetpassword',confirm=alert))
    except:
        alert = "ไม่สามารถส่งรหัส OTP ได้"
        return redirect(url_for('confirmForgetpassword',alert=alert))

@app.route("/checkOtp_forgetPassword",methods = ['post']) # ตรวจสอบ รหัส OTP จากการขอรหัสผ่าน
def checkOtp_forgetPassword():
    if request.method == "POST":
        user_otp = request.form['otp'] # OTP ที่ผู้ใช้กรอกเข้ามา
        gen_otp = session['gen_otp'] # OTP ที่ระบบสร้าง

        if gen_otp == user_otp :
            send_forgetpassword(session['email'],session['password'])

            session.clear()

            alert = "ระบบได้ทำการส่งรหัสผ่าน ไปยังอีเมลดังกล่าว"
            return redirect(url_for('confirmForgetpassword',confirm=alert))
        else:
            alert = "รหัส OTP ไม่ถูกต้อง"
            return redirect(url_for('confirmForgetpassword',alert=alert))                  
                    
@app.route("/home") # หน้าฟอร์ม home
def home():
    try:
        if session['email']:
            return render('home.html')
    except:
        return render('login.html')

@app.route("/camera",methods = ['post','get']) # หน้าเปิด กล้อง
def camera():
    try:
        if session['email']:
            alert_delay = get_delay(session['email'])[0]
            record_delay = get_delay(session['email'])[1]

            return render('camera.html',alert=alert_delay ,record=record_delay) # แสดงค่าการตั้งค่ากล้อง ของ ผู้ใช้
    except:
        return render('login.html')

@app.route("/videoFeed",methods = ['post','get'])# แสดง Video ไปยังหน้าฟอร์ม
def video_feed():
    alert_delay = get_delay(session['email'])[0]
    record_delay = get_delay(session['email'])[1]

    if get_linetoken(session['email']):
        line_token = get_linetoken(session['email'])[1]
    else:
        line_token = get_linetoken(session['email'])

    return Response(render_frame(session['email'],line_token,alert_delay,record_delay),mimetype='multipart/x-mixed-replace; boundary=frame')

def render_frame(user_email,line_token,alert_delay,record_delay):
    # ตัวแปรเริ่มต้น
    face_locations = []
    face_encodings = []

    record_status = 'off' # การบันทึกวิดิโอ
    record_timeStart = 0
    record_timeEnd = 0

    motion_foune = "no" # การตรวจจับการเคลี่อนไหว
    motionDetect_start = 0
    motionDetect_end = 0
    motionDetect_frame = None

    face_found = "no" # การตรวจจับใบหน้า
    faceDetect_start = 0
    faceDetect_end = 0

    fps = 0
    firstFrame = None
    firstFrame_state = 0
    alert_status = 'off'
    path_model = os.path.join('static',user_email,"Member","Training_Model","model.pkl")

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # เปลี่ยนสีภาพ
        blur = cv2.GaussianBlur(gray, (21, 21), 0) # เบลอภาพ

        # กำหนดภาพแรก สำหรับ การตรวจจับการเคลี่อนไหว
        if firstFrame is None:
            firstFrame_state += 1 

            if firstFrame_state == 60:
                firstFrame = blur
                firstFrame_state = 0 
            else:
                continue

        names = []
        alert_status = 'off'
        frame = cv2.putText(frame, datetime.now().strftime("%X"+" "+"%x"), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255 ,0), 2, cv2.LINE_8)

        if fps == 10:

            if face_found == 'no':
                face_locations = face_recognition.face_locations(frame)

                # ตรวจเจอใบหน้า
                if face_locations :
                    face_found = "yes"
                    face_encodings = face_recognition.face_encodings(frame, face_locations) # ถอดรหัสภาพใบหน้า 128-d
        
                    for face_encoding_unknows in face_encodings:

                        # เช็คไฟล์ โมเดล
                        if os.path.exists(path_model):
                            model = load(path_model)
                            confidence = model.predict_proba([face_encoding_unknows])
                            print(model.predict([face_encoding_unknows]),np.ndarray.max(confidence))

                            # เช็คค่าความแม่นยำ ของ การเปรียบเทียวใบหน้า
                            if (np.ndarray.max(confidence)) >= 0.7 :
                                name = model.predict([face_encoding_unknows])
                                # print('ตรวจพบใบหน้า '+name[0])

                            else:
                                name = ['Unknows']
                                alert_status = 'on'

                                faceUnknow_image = time_now() + ".jpg"
                                cap_frame(frame,faceUnknow_image,user_email,line_token)
                                # print('ตรวจพบใบหน้าที่ไม่รู้จัก')

                        else:
                            name = ['Unknows']
                            alert_status = 'on'

                            faceUnknow_image = time_now() + ".jpg"
                            cap_frame(frame,faceUnknow_image,user_email,line_token)
                            # print('ตรวจพบใบหน้าที่ไม่รู้จัก')

                        names.append(name)
                    
                    # วาดกรอบ และ ชื่อ
                    for (top, right, bottom, left), name in zip(face_locations, names):
                        if name[0] == 'Unknows':
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        else:
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                        cv2.putText(frame, name[0], (left + 3, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

                    motionDetect_start = 0
                    motionDetect_end = 0

                    motion_foune = "no"

                    faceDetect_start = time.time()
                    faceDetect_end = time.time()
                    
                else: 
                    face_found = "no"

                # ตรวจเจอการเคลี่อนไหว
                if (face_found == "no") and (motion_foune == 'no'):
                    frameDelta = cv2.absdiff(firstFrame, blur)
                    thresh = cv2.dilate(cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
                    cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for c in cnts:
                        if cv2.contourArea(c) < 5000:
                            continue

                        motionDetect_start = time.time()
                        motionDetect_end = time.time()
                        motion_foune = "yes"

                        motionDetect_frame = frame
                        # print('ตรวจพบการเคลื่อนไหว')

                elif (motion_foune == 'yes') and ((motionDetect_end - motionDetect_start) > 5):
                    motionDetect_start = 0
                    motionDetect_end = 0
                    motion_foune = "no"

                    alert_status = "on"

                    motionDetect_name = time_now() + ".jpg"
                    cap_frame(motionDetect_frame,motionDetect_name,user_email,line_token)
                    motionDetect_frame = None
                    
                    # print('แจ้งเตือนการเคลี่อนไหว')

                elif (motion_foune == 'yes'):
                    motionDetect_end = time.time()
                    # print('ดีเลการเตลี่อนไหว')
                    
            elif face_found == 'yes' and ((faceDetect_end - faceDetect_start) > alert_delay) :
                face_found = 'no'
                alert_status = 'off'

            elif face_found == 'yes':
                faceDetect_end = time.time()
                # print('ดีเลใบหน้า')

            # บันทึกวิดิโอ
            if record_status == 'off' and alert_status == "on":
                video_name = time_now() + ".mp4"
                video_path = os.path.join(os.getcwd(),'static',user_email,'Video',(video_name))

                out = cv2.VideoWriter(video_path,fourcc, 20.0, (640,480))
                    
                record_status = 'on'
                record_timeStart = time.time()
                record_timeEnd = time.time()
                t1 = datetime.now().strftime("%X")

                # print('เริ่มบันทึกวิดิโอ')

            elif (record_timeEnd - record_timeStart) > record_delay:
                out.release()

                record_status = 'off'
                record_timeStart = 0
                record_timeEnd = 0
                t0 = datetime.now().strftime("%X")

                cur = conn.cursor()
                sql = "INSERT INTO videos_member (videos_name,videos_date,videos_timeStart,videos_timeEnd,member_email) VALUES (%s,%s,%s,%s,%s)" # เพิ่ม ข้อมูลการแจ้งเตือนใน Database
                cur.execute(sql,(video_name,datetime.now().strftime("%d"+"/"+"%m"+"/"+"%y"),t1,t0,user_email))
                conn.commit()
                cur.close()

                # print('สิ้นสุดการบันทึกวิดิโอ')

            elif record_status == 'on':
                out.write(frame)

                record_timeEnd = time.time()

                # print('กำลังบันทึกวิดิโอ')

            fps = 0
            
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg',frame)[1].tobytes() + b'\r\n')
                
        else:
            fps += 1
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg',frame)[1].tobytes() + b'\r\n')
            
def cap_frame(frame,image_name,user_email,line_token): # บันทึกรูปใบหน้าที่ตรวจพบ
    image_path = os.path.join(os.getcwd(),'static',user_email,'Alert',(image_name))
    try:
        cv2.imwrite(image_path,frame) # เพิ่ม รูปลงในโฟลเดอร์ Alert

        cur = conn.cursor()
        sql = "INSERT INTO alerts_member (alert_name,alert_date,alert_time,member_email) VALUES (%s,%s,%s,%s)" # เพิ่ม ข้อมูลการแจ้งเตือนใน Database
        cur.execute(sql,(image_name,datetime.now().strftime("%d"+"/"+"%m"+"/"+"%y"),datetime.now().strftime("%X"),user_email))
        conn.commit()
        cur.close()

        send_image(datetime.now().strftime("%X"+" "+"%x"), image_path, line_token)# ส่ง รูปและเวลา ไปทาง Line
        
    except:
        pass

@app.route("/settingCamera",methods = ['post']) # บันทึก การตั้งค่ากล้อง
def setting():
    if request.method == "POST":
        alert_delay = request.form['alert_delay']
        record_delay = request.form['record_delay']

        cur = conn.cursor()
        sql = "UPDATE register_member SET delay_alert = %s , delay_record = %s WHERE member_email = %s"
        cur.execute(sql,(alert_delay,record_delay,session['email']))
        conn.commit()
        cur.close()

    return redirect(url_for('camera'))

@app.route("/videoGallery",methods= ['post','get']) # หน้าฟอร์ม คลังวิดิโอ
def videoGallery(): 
    try:
        if session['email']:
            cur = conn.cursor()
            sql = 'SELECT videos_name,videos_date,videos_timeStart,videos_timeEnd FROM videos_member WHERE member_email = %s'
            cur.execute(sql,(session['email'])) 
            rows = cur.fetchall()
            cur.close()

            name_list = [] # สร้าง list เก็บข้อมูล ชื่อวิดิโอ ที่ SELECT ออกมา
            date_list = [] # สร้าง list เก็บข้อมูล วันวิดิโอ ที่ SELECT ออกมา
            for row in rows:
                name_list.append(row[0]) # row[0] = ข้อมูล ชื่อ
                date_list.append(row[1]) # row[1] = ข้อมูล วัน

            # ลบ ข้อมูลที่ซ้ำกันใน List
            name_list = list(dict.fromkeys(name_list))   
            date_list = list(dict.fromkeys(date_list))

            # ลบไฟล์ที่เกิดจากบัค
            file_names = glob.glob1(os.path.join('static',session['email'],'Video'),'*') 
            for file_name in file_names:
                if file_name in name_list:
                    pass
                else:         
                    os.remove(os.path.join(os.getcwd(),'static',session['email'],"Video",file_name))

            return render('video_gallery.html',dates=date_list ,video_datas=rows)
    except:
        return render('login.html')

@app.route("/deleteVideo",methods= ['post']) # ลบ คลังวิดิโอ
def deleteVideo(): 
    try:
        if session['email']:
            video_name = request.form['video_name']
            os.remove(os.path.join('static',session['email'],'Video',video_name)) # ลบ ข้อมูลออกจาก โฟลเดอร์

            cur = conn.cursor()
            sql = 'DELETE FROM videos_member WHERE videos_name = %s' # ลบ ข้อมูลออกจาก Database
            cur.execute(sql,(video_name)) 
            conn.commit()
            cur.close()

            return redirect(url_for('videoGallery'))
    except:
        return render('login.html')

@app.route("/addNotification",methods = ['post']) # แก้ไข line token
def addNotification():
    if request.method == "POST":
        notification_name = request.form['notification_name']
        notification_linetoken = request.form['notification_linetoken']

        cur = conn.cursor()
        sql = 'INSERT INTO linetoken_member (token_name,token_line,member_email) VALUES (%s,%s,%s)'
        cur.execute(sql,(notification_name,notification_linetoken,session['email']))
        conn.commit()
        cur.close()

    return redirect(url_for('history'))

@app.route("/editNotification",methods = ['post']) # แก้ไข line token
def editNotification():
    if request.method == "POST":
        if request.form['submit'] == 'selete':
            token_id = request.form['token_id']

            # แก้ไขค่า ใน data base
            cur = conn.cursor()
            sql = 'UPDATE register_member SET token_id = %s WHERE member_email = %s'
            cur.execute(sql,(token_id,session['email']))
            conn.commit()
            cur.close()

        if request.form['submit'] == 'delete':
            token_id = request.form['token_id']

            # ลบค่า ใน data base
            cur = conn.cursor()
            sql = 'DELETE FROM linetoken_member WHERE token_id = %s'
            cur.execute(sql,(token_id))

            sql = 'UPDATE register_member SET token_id = %s WHERE member_email = %s'
            cur.execute(sql,('0',session['email']))
            conn.commit()
            cur.close()

    return redirect(url_for('history'))

@app.route("/notificationHistory") # หน้าฟอร์ม ประวัติการแจ้งเตือน
def history():
    try:
        if session['email'] :

             # ดึงข้อมูลช่วง วัน เวลา และ ชื่อ ของการแจ้งเตือน
            cur = conn.cursor()
            sql = 'SELECT alert_name,alert_date,alert_time FROM alerts_member WHERE member_email = %s'
            cur.execute(sql,(session['email'])) 
            datetimes = cur.fetchall()
                
            date_list = [] # สร้าง list เก็บข้อมูล วัน ที่ SELECT ออกมา
            time_list = [] # สร้าง list เก็บข้อมูล เวลา ที่ SELECT ออกมา

            for datetime in datetimes:
                date_list.append(datetime[1]) # datetime[1] = ข้อมูล วัน
                time_list.append(datetime[2]) # datetime[2] = ข้อมูล เวลา

            date_list = list(dict.fromkeys(date_list)) # ลบ ข้อมูลที่ซ้ำกันใน List

            # ดึงข้อมูล line token
            cur = conn.cursor()
            sql = 'SELECT token_id,token_name FROM linetoken_member WHERE member_email = %s'
            cur.execute(sql,(session['email'])) 
            data_token = cur.fetchall()
            cur.close()

            if get_linetoken(session['email']):
                user_token = get_linetoken(session['email'])[0]
            else:
                user_token = get_linetoken(session['email'])

            if len(data_token) == 0:
                alert = 'กรุณาเพิ่มช่องทางสำหรับการแจ้งเตือน'
                return render('notification_history.html',images=datetimes ,dates=date_list ,data_token=data_token ,user_token=user_token ,alert=alert)

            return render('notification_history.html',images=datetimes ,dates=date_list ,data_token=data_token ,user_token=user_token)
    except:
        return render('login.html')

@app.route("/notificationHistory",methods = ['post']) # ลบ ประวัติการแจ้งเตือน
def deleteHistory():
    if request.method == "POST":
        img = request.form['img_alert']

        os.remove(os.path.join('static',session['email'],'Alert',img)) # ลบ ข้อมูลออกจาก โฟลเดอร์

        cur = conn.cursor()
        sql = 'DELETE FROM alerts_member WHERE alert_name = %s' # ลบ ข้อมูลออกจาก Database
        cur.execute(sql,(img)) 
        conn.commit()
        cur.close()

    return redirect(url_for('history'))
    
@app.route("/member") # หน้าฟอร์ม แก้ไขสมาชิก
def member():
    try:
        if session['email'] :
            cur = conn.cursor()
            sql = 'SELECT image_folder,image_count FROM images_member WHERE member_email = %s'
            cur.execute(sql,(session['email'])) 
            rows = cur.fetchall()
            cur.close()

            folder_list = []
            for row in rows:
                folder_list.append(row[0])
            folder_list = list(dict.fromkeys(folder_list))

            # ลบไฟล์ที่เกิดจากบัค
            file_names = glob.glob1(os.path.join('static',session['email'],'Member'),'*') 
            for file_name in file_names:
                if file_name in folder_list or file_name == 'Training_Model' :
                    pass
                else:         
                    shutil.rmtree(os.path.join(os.getcwd(),'static',session['email'],"Member",file_name))

            path_model = os.path.join('static',session['email'],"Member","Training_Model","model.pkl")
            
            if os.path.exists(path_model):
                return render('member.html' ,folder=rows)
            else:
                alert = "เพิ่มสมาชิกอย่างน้อย 2 คนขึ้นไป"
                return render('member.html' ,folder=rows ,alert=alert) # แสดงชื่อ กับ รูป ของสมาชิก folder[0] = ชื่อโฟล์เดอร์ , folder[1] = ชื่อรูปภาพ
    except:
        return render('login.html')

@app.route("/addMember",methods = ['post']) # เพิ่ม สมาชิก
def addMember():
    if request.method == "POST":
        folder_name = request.form['name']
        path_images = request.files.getlist('add_image')

        path_model = os.path.join('static' ,session['email'] ,"Member" ,"Training_Model" ,"model.pkl")
        path = os.path.join(os.getcwd(),'static',session['email'],'Member',folder_name) # เปลี่ยน path ไปยัง โฟล์เดอร์ของสมาชิกที่ login เข้ามา

        try:
            os.mkdir(path)
            os.mkdir(os.path.join(path,'Profile'))

            for image in path_images: # แสดงชื่อ รูปทีละรูป
                image.save(os.path.join(path,secure_filename(image.filename))) # บันทึกภาพที่เลือกไว้ในโฟล์เดอร์ สมาชิก

                count = len(glob.glob(os.path.join(os.getcwd(),'static',session['email'],'Member',folder_name,'*.jpg')))
                if count == 1:# ทำรูปโปรไฟล์
                    face_detection(os.path.join(path,secure_filename(image.filename)),folder_name,session['email'])

                old_file_name = os.path.join(path,secure_filename(image.filename))
                new_file_name = os.path.join(path,folder_name + str(count) + '.jpg')
                
                os.rename(old_file_name, new_file_name)

            cur = conn.cursor()
            sql = "INSERT INTO images_member (image_folder,image_count,member_email) VALUES (%s,%s,%s)" # เพิ่ม ข้อมูลสมาชิกใน Database
            cur.execute(sql,(folder_name,count,session['email']))
            conn.commit()
            cur.close()

            try:
                if os.path.exists(path_model):
                    os.remove(path_model)
                    create_model(session['email'])
                else:
                    create_model(session['email'])
            except:
                pass

            alert = "เพิ่มข้อมูลสมาชิกเรียบร้อย"
            return redirect(url_for('member' ,confirm=alert))

        except:
            alert = "ไม่สามารถเพิ่มข้อมูลสมาชิกนี้ได้"
            return redirect(url_for('member',alert=alert))

def face_detection(image_path,folder_name,user_email): # ตัดเอาเฉพาะภาพใบหน้า
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_dataset.xml')
    img = cv2.imread(image_path)
    
    faces = face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))

    height, width = img.shape[:2]

    for (x, y, w, h) in faces:
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)
        
        faceimg = img[ny:ny+nr, nx:nx+nr]
        lastimg = cv2.resize(faceimg, (50,50))
        cv2.imwrite(os.path.join(os.getcwd(),'static',user_email,'Member',folder_name,'profile',folder_name+'.jpg') , lastimg)

@app.route("/deleteMember",methods = ['post']) # ลบ สมาชิก
def editMember():
    if request.method == "POST":
        if request.form['submit'] == 'delete':
            name = request.form['name']

            path_model = os.path.join('static' ,session['email'] ,"Member" ,"Training_Model" ,"model.pkl")

            shutil.rmtree(os.path.join(os.getcwd(),'static',session['email'],"Member",name)) # ลบ โฟล์เดอร์สมาชิก
            
            cur = conn.cursor()
            sql = 'DELETE FROM images_member WHERE image_folder = %s' # ลบ ข้อมูลสมาชิกออกจาก Database
            cur.execute(sql,(name)) 
            conn.commit() 
            cur.close()

            try:
                if os.path.exists(path_model):
                    os.remove(path_model)
                    create_model(session['email'])
                else:
                    create_model(session['email'])
            except:
                pass

            alert = "ลบข้อมูลสมาชิกเรียบร้อย"
            return redirect(url_for('member' ,confirm=alert))

        if request.form['submit'] == 'edit':
            folder_name = request.form['name']
            path_images = request.files.getlist('edit_image')

            if request.files['edit_image'].filename == '':
                alert = "โปรดกรอกข้อมูลให้ครบถ้วน"
                return redirect(url_for('member',alert=alert))

            path_model = os.path.join('static' ,session['email'] ,"Member" ,"Training_Model" ,"model.pkl")
            path = os.path.join(os.getcwd(),'static',session['email'],'Member',folder_name) # เปลี่ยน path ไปยัง โฟล์เดอร์ของสมาชิกที่ login เข้ามา

            try:
                for image in path_images: # แสดงชื่อ รูปทีละรูป
                    image.save(os.path.join(path,secure_filename(image.filename))) # บันทึกภาพที่เลือกไว้ในโฟล์เดอร์ สมาชิก
                    
                    # เปลี่ยนชื่อภาพ
                    count = len(glob.glob(os.path.join(os.getcwd(),'static',session['email'],'Member',folder_name,'*.jpg'))) # นำจำนวนภาพในโฟลเดอร์
                    old_file_name = os.path.join(path,secure_filename(image.filename)) # ชื่อไฟล์เก่า
                    new_file_name = os.path.join(path,folder_name + str(count) + '.jpg') # ชื่อไฟล์เก่า
                
                    os.rename(old_file_name, new_file_name) # เปลี่ยนชื่อภาพ
                
                cur = conn.cursor()
                sql = 'UPDATE images_member SET image_count = %s WHERE image_folder = %s' #  แก้ไข ข้อมูล รูปสมาชิก จาก Database
                cur.execute(sql,(count,folder_name)) 
                conn.commit() 
                cur.close()

                try:
                    if os.path.exists(path_model):
                        os.remove(path_model)
                        create_model(session['email'])
                    else:
                        create_model(session['email'])
                except:
                    pass

                alert = "อัปเดตข้อมูลสมาชิกเรียบร้อย"
                return redirect(url_for('member' ,confirm=alert))

            except:
                alert = "ไม่สามารถอัปเดตข้อมูลสมาชิกนี้ได้"
                return redirect(url_for('member',alert=alert))

@app.route("/admin") # หน้าฟอร์มสำหรับ จัดการสมาชิก
def admin():
    cur = conn.cursor()
    sql = 'SELECT * FROM register_member'
    cur.execute(sql) 
    rows = cur.fetchall()
    cur.close()

    return render('admin.html',datas = rows)

@app.route("/admin",methods = ['post']) # ลบสมาชิก ออกจาก database
def manage():
    if request.method == "POST":
        member_email = request.form['member_email']
        cur = conn.cursor()

        sql = 'DELETE FROM register_member WHERE member_email = %s' # ลบข้อมูลทั้งหมดใน register_member
        cur.execute(sql,(member_email))
        sql = 'DELETE FROM images_member WHERE member_email = %s' # ลบข้อมูลทั้งหมดใน images_member
        cur.execute(sql,(member_email))
        sql = 'DELETE FROM alerts_member WHERE member_email = %s' # ลบข้อมูลทั้งหมดใน alerts_member
        cur.execute(sql,(member_email))
        sql = 'DELETE FROM videos_member WHERE member_email = %s' # ลบข้อมูลทั้งหมดใน videos_member
        cur.execute(sql,(member_email))
        sql = 'DELETE FROM linetoken_member WHERE member_email = %s' # ลบข้อมูลทั้งหมดใน linetoken_member
        cur.execute(sql,(member_email))

        conn.commit()
        cur.close()

        shutil.rmtree(os.path.join(os.getcwd(),'static',member_email))

    return redirect(url_for('admin'))

if __name__ == "__main__":
    app.run(debug=True)
