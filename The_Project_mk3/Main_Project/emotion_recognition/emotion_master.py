"""
Project: emotion_recognition
작성자: 김민관, 박세빈
프로그램 내용
 - 이 프로그램은 카메라와 마이크로 사람의 감정을 인식하는 프로그램이다
 - facial_emotion 과 word_emotion 으로 나누어서 작동한다
 - 각각의 파일에서 index를 txt형태로 받아서 가지고 온다
 - 계산된 데이터는 csv파일로 바꾸어 저장된다.
 - 하위 파일에 대한 설명을 하위 파일에 명시되어있다.
"""

"""
version 1.0  - faical_emotion.py 변경 
version 1.1  - 7가지 감정에서 긍정 부정 2가지로 변경됨
version 1.2  - word_emotion작업중 내용 변경
latest version : 1.2
작성자 : 김민관, 박세빈
"""

from facial_emotion import facial_emotion_detector as FED
from word_emotion import inference
import emotion_score
from threading import Thread
import time

flag = 0

def emotion_control(): 
    print("word_emotion inference start")
    newThread = Thread(target = inference.word_inference) #sub thread work
    newThread.start()

    print("facial emotion detector start") #main thread work
    FED.detector() #facial emotion detector start

    while True:
        if (flag == 1): #if main thread return, call sub thread to stop
            break
        else: #or main thread need to wait
            time.sleep(1)
            
    
    emotion_score.makeResponse()

    return