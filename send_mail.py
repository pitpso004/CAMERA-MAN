import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os.path

def send_forgetpassword(User_email,User_password,attachment_location=''):

    msg = MIMEMultipart()

    msg['From'] = 'CAMERA MAN' # ชื่อผู้ส่ง
    msg['To'] = User_email # mail ผู้รับ
    msg['Subject'] = "ลืมรหัสผ่าน" # หัวข้อ ที่ส่ง

    body = "รหัสผ่านของ Email: " + User_email +" คือ "+User_password # เนื้อหา ที่ส่ง

    msg.attach(MIMEText(body,'plain'))

    if attachment_location != '':
        filename = os.path.basename(attachment_location)
        attachment = open(attachment_location, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        "attachment; filename= %s" % filename)
        msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login("n4o0n42541@gmail.com", "n4122541")
        text = msg.as_string()
        server.sendmail('CAMERA MAN', User_email, text)
        print('email sent successfully')
        server.quit()
    except:
        print("SMPT server connection error")
    return True

def send_otp(User_email,OTP,Link,attachment_location=''):

    msg = MIMEMultipart()

    msg['From'] = 'CAMERA MAN' # ชื่อผู้ส่ง
    msg['To'] = User_email # mail ผู้รับ
    msg['Subject'] = "ยืนยันอีเมล" # หัวข้อ ที่ส่ง

    body = "รหัส OTP ของ Email: " + User_email +" คือ "+OTP+"\n" # เนื้อหา ที่ส่ง
    body = body + "ลิ้งยืนยันอีเมล http://127.0.0.1:5000/"+Link

    msg.attach(MIMEText(body,'plain'))

    if attachment_location != '':
        filename = os.path.basename(attachment_location)
        attachment = open(attachment_location, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        "attachment; filename= %s" % filename)
        msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login("n4o0n42541@gmail.com", "n4122541")
        text = msg.as_string()
        server.sendmail('CAMERA MAN', User_email, text)
        print('email sent successfully')
        server.quit()
    except:
        print("SMPT server connection error")
    return True

# send_otp('pitpso004@hotmail.com','1234','www.google.com')