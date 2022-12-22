from flask import Flask, request, jsonify
import time
import os
import sys

app = Flask(__name__) #flask server start
local_path=os.path.dirname(os.path.realpath(__file__)) #this file path
emotion_file = "emotion_detector.py" #emotion detector file name
#GPIO setting here

#test code --------------------------------
@app.route('/keyboard') 
def keyboard():
    dataSend = {
        "type" : "text"
    }
#if trying to test this project, send message from kakaotalk

def kakaoService(emotion_data):
    trigger = 0
    app.run(host="", port = 5000, debug =True)
    @app.route('/message', methods=['post']) #message main function start
    def mssage(): #main function
        dataRecive = request.get_json()
        content = dataReceive['content']

        if content == "감정분석":
            dataSend = {
                "message" : {
                    "text" : "감정분석을 시작합니다."
                }
            }
            cmd = "sudo python3 " + local_path + "/" + emotion_file + " &"
            print("cmd: ", cmd)
            os.system(cmd)           
            time.sleep(0.5)

        if emotion_data == 1:
                dataSend = {
                "message" : {
                    "text" : "현재 사용자의 감정은 '긍정'입니다."
                }
            }

        elif emotion_data == -1:
                dataSend = {
                "message" : {
                    "text" : "현재 사용자의 감정은 '부정'입니다."
                }
            }

        elif emotion_data == 0:
                dataSend = {
                "message" : {
                    "text" : "현재 사용자의 감정은 '특징없음'입니다."
                }
            }

        return jsonify(dataSend)

if __name__ == "__main__":
    kakaoService()