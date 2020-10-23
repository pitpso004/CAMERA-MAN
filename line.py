import requests
import os

def send_line(message,user_token):
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization':'Bearer '+ user_token}
    r = requests.post(url , headers=headers , data = {'message':message})

def send_image(time,message,user_token):
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization':'Bearer '+ user_token}
    r = requests.post(url , headers=headers , data = {'message':time} , files={'imageFile':open(message,'rb')})



