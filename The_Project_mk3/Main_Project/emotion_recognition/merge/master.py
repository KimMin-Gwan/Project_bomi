"""
file name : emotion_detector.py
details : 감정인싱의 거의 모든 기능을 제어하는 master(진)파일  
version 1.0  -  emotion_detector.py 작성
version 1.1  -  random으로 출력값 지정 
version 1.2  -  약 복용, 식사 안부 추가
version 1.3  -  랜덤 농담 및 보미 생일(이스터에그) 추가
version 1.4  -  오늘 날자 말하기 추가
version 1.5  -  sh파일로 실행시 경로지정 오류 문제 해결
version 1.6  -  카메라 인식 일정시간 후 꺼짐 기능 구현 (미완 추가예정)
version 1.7  -  감정에 따른 피드백 추가
version 1.8  -  약 복용시간 확인 추가
lastest version : 1.8
작성자 : 김민관, 박세빈
"""

import os
from tempfile import tempdir
import cv2
import numpy as np
from keras.models import model_from_json
import tensorflow as tf
import time #주기적인 트리거 실행을 위한 모듈
import sys
import pandas as pd 
from playsound import playsound #mp3파일을 재생하기 위한 모듈
import re
import json #학습한 데이터를 가져오기 위해 사용
from konlpy.tag import Okt
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle #인덱스 저장을 위해 사용
import keras #머신러닝 모델을 활용하기 위한 모듈
import random #Feedback 값 랜덤 출력을 위한 모듈
import schedule #주기적인 트리거 실행을 위한 모듈
from gtts import gTTS #gtts 모듈



meal_time_breakfast_hour = "07" #아침 식사 시간 (시)
meal_time_launch_hour = "12" #점심 식사 시간 (시)
meal_time_dinner_hour = "18" #저녁 식사 시간 (시)
meal_time_min = random.randint(45, 59) #식사시간 랜덤 생성 (분)

rand_min = random.randint(10, 59) #농담 던지는 랜덤 time(분)

"""
3번의 농담을 던지는 겹치지 않는 시간 생성 후 리스트에 넣기
"""
#-----------------농담-----------------------
joke_time = []

rand_num = (random.randint(10, 17)) #10시부터 17시까지 랜덤 실행


for i in range(3):
    
    # 현재 생성된 랜덤 숫자가 이미 numbers 리스트에 존재하면 다시 생성 (중복방지)
    while rand_num in joke_time:
        rand_num = (random.randint(10, 21))
    # while 문을 벗어났단 이야기는 중복된 숫자가 아니니 numbers 리스트에 추가
    joke_time.append(rand_num)
#---------------------------------------------

tm = time.localtime() #현재 시간 


facial_result = 0 #결과 글로벌 함수 지정 (얼굴)
word_result = 0 #결과 글로벌 함수 지정 (음성)
emotion_tier = 0

random_index = random.randint(1,3) #1이상 3이하의 정수 리턴 (감정 표현 랜덤 출력)

model_dir = os.path.dirname(os.path.realpath(__file__)) #자동 실행시 경로 지정 오류 해결 (절대 경로 지정)
model_name1 = "fer.json"
model_name2 = 'fer.h5'

Full_model1_dir = os.path.join(model_dir, model_name1)
Full_model2_dir = os.path.join(model_dir, model_name2)

#load model
model = model_from_json(open(Full_model1_dir, "r").read())
#model = model_from_json(open("fer.json").read())
#load weights
model.load_weights(Full_model2_dir)

okt = Okt()
tokenizer  = Tokenizer()

DATA_CONFIGS = './CLEAN_DATA/data_configs.json'
DATA_Con_dir = os.path.join(model_dir, DATA_CONFIGS)

prepro_configs = json.load(open(DATA_Con_dir,'r', encoding = 'cp949')) #TODO 데이터 경로 설정

#TODO 데이터 경로 설정
Tokens = './CLEAN_DATA/tokenizer.pickle'
Token_dir = os.path.join(model_dir, Tokens)
with open(Token_dir,'rb') as handle:
    word_vocab = pickle.load(handle)

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')


cap=cv2.VideoCapture(0)

"""
file name : facial_detector.py
details : 카메라를 통해 얼굴을 촬영하고 감정데이터를 분석해 긍정과 부정으로  
version 1.0  -  facial_detector.py 작성
version 1.1  -  흑백으로 꺼지지 않는 버그 수정 (카메라 출력 x)
version 1.2  -  그럼에도 꺼지지 않는 버그 수정 (카메라 꺼버림)
lastest version : 1.2
작성자 : 김민관, 박세빈
"""
        
def facial_detector(): #얼굴인식
    global facial_result
    playsound(os.path.join(model_dir,"face_detect_start.mp3"))
    safe_zone = 5
    index = 0
    i = 0
    capture = 0
    while True:
        ret,test_img=cap.read()# captures frame and returns boolean value and captured image
        if not ret:
            continue
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        
        if i == 10:
            break
        
        for (x,y,w,h) in faces_detected:
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
            roi_gray=cv2.resize(roi_gray,(48,48))
            img_pixels = tf.keras.utils.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            #img_pixels /= 255

            predictions = model.predict(img_pixels)

            #find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            if(max_index == 0): #angry
                capture = capture - 1
            elif(max_index == 1):   #disgust
                capture = capture - 1
            elif(max_index == 2):   #fear
                capture = capture - 1
            elif(max_index == 3):   #happy
                capture = capture + 2
            elif(max_index == 4):   #sad
                capture = capture - 1
                
            elif(max_index == 5):   #surprise
                capture = capture + 1
            i = i + 1
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        #cv2.imshow('Facial emotion analysis ',resized_img) #카메라 출력
        if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
            break

    if(abs(capture) < safe_zone):
        index = 0
    elif(capture > safe_zone):
        index = 1
    else:
        index = -1
    facial_result = index                    
    """
    f = open('emotion_reuslt.txt', 'w')
    print(index, file=f)
    f.close()
    """
    #cv2.VideoCapture.release()
    #cap.release()
    #cv2.destroyAllWindows()
    #time.sleep(1)
    
    return

"""
file name : inference.py
details : txt로 받아온 음성인식 데이터를 긍정과 부정으로 구분
 - 정확도가 65를 넘어갈때만 인정 그외는 모두 의미없는 데이터로 취급
 - emotion_result.txt에 저장 하여 전달

version 1.0  -  inference.py 작성
version 1.1  -  파일 입출력 수정
version 1.2  -  3보다 작은 값 입력시 다시입력 
lastest version : 1.2
작성자 : 김민관, 박세빈
"""


def word_inference():
    playsound(os.path.join(model_dir,"voice_input.mp3")) #음성인식을 시작합니다. 오늘 하루는 어떠셨나요?
    time.sleep(8)
    global word_result

    prepro_configs['vocab'] = word_vocab

    tokenizer.fit_on_texts(word_vocab)

    MAX_LENGTH = 8 #문장최대길이
    
    
    fp = open('/home/yuice2/envbomi/emotion_input.txt', 'r') #Google Assistance에서 받아올 값 파일로 open
    sentence = fp.readlines()[-1] #sentence에 값 저장, 마지막 줄 값 읽어오기, 정상 작동 확인
    fp.close()
    print("입력된 단어 : ", sentence)

    """
    while True:
        fp = open('/home/yuice2/envbomi/emotion_input.txt', 'r') #Google Assistance에서 받아올 값 파일로 open
        sentence = fp.readlines()[-1] #sentence에 값 저장, 마지막 줄 값 읽어오기, 정상 작동 확인
        fp.close()
        print("입력된 단어 : ", sentence)
        if len(sentence) < 3:
            playsound("voice_fail.mp3")
            time.sleep(1)
            
        else:
            break
    """
    sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\\s ]','', sentence)
    stopwords = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한'] # 불용어 추가할 것이 있으면 이곳에 추가
    sentence = okt.morphs(sentence, stem=True) # 토큰화
    sentence = [word for word in sentence if not word in stopwords] # 불용어 제거
    vector  = tokenizer.texts_to_sequences(sentence)
    pad_new = pad_sequences(vector, maxlen = MAX_LENGTH) # 패딩

    #학습한 모델 불러오기
    model = keras.models.load_model('./my_models/') #TODO 데이터 경로 설정
    model.load_weights('./DATA_OUT/cnn_classifier_kr/weights.h5') #TODO 데이터 경로 설정
    predictions = model.predict(pad_new)
    predictions = float(predictions.squeeze(-1)[1])

    if(predictions > 0.5):
        temp = predictions * 100
        if(temp < 65):
            index = 0  #nomal result (meanless data)
        else:
            index = 1  #negative result
        """
        f = open('emotion_reuslt.txt', 'w')
        print(index, file=f)
        f.close()
        """
    else:
        temp = (1 - predictions) * 100
        if(temp < 65):
            index = 0 #nomal result (meanless data)
        else:
            index = -1  #negative result
        """
        f = open('emotion_reuslt.txt', 'w')
        print(index, file=f)
        f.close()
        """
    word_result= index

    return

"""
Project: emotion_score
작성자: 박세빈
프로그램 내용
 - 이 프로그램은 카메라와 마이크로 사람의 감정을 인식하는 프로그램이다
 - facial_emotion 과 word_emotion 으로 나누어서 작동한다
 - 각각의 파일에서 index를 txt형태로 받아서 가지고 온다
 - 계산된 데이터는 txt파일로 저장된다.
 - 하위 파일에 대한 설명을 하위 파일에 명시되어있다.
"""

"""
version 1.0  - faical_emotion.py 변경
version 1.1  - 7가지 감정에서 긍정 부정 2가지로 변경됨
version 1.2  - word_emotion작업중 내용 변경
version 1.3  - 계산된값 txt파일로 저장
version 1.4  - 감정표현 및 mp3파일 추가
latest version : 1.4
작성자 : 박세빈
pip install playsound==1.2.2
"""



#from stringprep import map_table_b3
#from facial_emotion import facial_emotion_detector as face
#from word_emotion import word_emotion_detoctor as word
    

def emotion_data():
    global facial_result, word_result
    global emotion_tier #현재 상태 저장
    #emotion_recognizer_face.py를 실행하여 표정 데이터를 받는다.
    #emotion_recognizer_word.py를 실행하여 감정 데이터를 받는다.
    #감정데이터 작성이 끝나면 아래에서 감정 데이터를 받아온다.
    #두가지 데이터를  3:6의 비율로 계산한다. 크기는 100
    #감정 데이터를 pandas를 이용하여 .csv파일로 저장한다.
    #.csv파일을 바탕으로 감정 기복을 확인한다.
    #부정적인 감정이 지속되면 유튜브로 음악을 틀어주거나 한다.

    #인식된 데이터를 받아옴-------------------------------------------------------------
    """
    f1 = open('./word_emotion/emotion_result.txt', 'r')
    f2 = open('./facial_emotion/emotion_result.txt', 'r')
    word_data = f1.read()
    face_data = f2.read()
    f1.close()
    f2.close() #정상출력
    """

    #얼굴인식 점수와 음성인식 점수를 가져와 더함-------------------------------------------
    word_data = int(word_result) * 10
    face_data = int(facial_result) * 10 #값을 더하기 위한 형변환
    result = (word_data * 7 / 10) + (face_data * 3 / 10) #얼굴인식 점수와 음성인식 점수 비율 3 : 7

    #결과값을 emotion_result파일에 한줄 씩 저장
    
    f = open('emotion_data.txt', 'a')
    f.write(str(result) + '\n')
    f.close()
    f = open('emotion_data.txt', 'r')
    saved_data = f.read()
    f.close()
    lastFiveData = saved_data.split("\n")
    lastFiveData = lastFiveData[-2:-1] #마지막 공백 제외 뒤에서 2개의 값 가져옴 
    
    #마지막 5번의 데이터를 분석함 ------------------------------------------------------
    # 최근 마지막 5개의 데이터에서 긍정적인 데이터가 많다면 긍정적인 반응을
    # 부정적인 데이터가 많다면 노래를 틀어주도록 함 
    
    sum = 0
    
    for sampleValue in lastFiveData: #마지막 2개 데이터 더하기
        sum = float(sampleValue) + sum
    print(sum)

    if(sum < 0):    #부정
        
        print("negative")
        emotion_tier = -1 #현재상태 -1(부정)
        if(random_index == 1):
            playsound(os.path.join(model_dir,"feeling_bad.mp3"))
            #playsound("Daddy.mp3") 
        elif(random_index == 2):
            playsound(os.path.join(model_dir,"feeling_bad2.mp3"))
        elif(random_index == 3):
            playsound(os.path.join(model_dir,"feeling_bad3.mp3"))
        else:
            print("오류가 발생했습니다")

    elif(sum > 0):  #긍정
        print("positive")
        emotion_tier = 1 #현재상태 1(긍정)
        if(random_index == 1):
            playsound(os.path.join(model_dir,"feeling_happy.mp3"))
        elif(random_index == 2):
            playsound(os.path.join(model_dir,"feeling_happy2.mp3"))
        elif(random_index == 3):
            playsound(os.path.join(model_dir,"feeling_happy3.mp3"))
        else:
            print("오류가 발생했습니다")
        
    else:   #중립
        print("nothing")
        emotion_tier = 0 #현재상태 0 (중립)
        if(random_index == 1):
            playsound(os.path.join(model_dir,"feeling_neutral.mp3"))
            #playsound(os.path.join(model_dir,"AmorParty.mp3"))

        elif(random_index == 2):
            playsound(os.path.join(model_dir,"feeling_neutral2.mp3"))
        elif(random_index == 3):
            playsound(os.path.join(model_dir,"feeling_neutral3.mp3"))
        else:
            print("오류가 발생했습니다")

'''
    print("main start")
    schedule.every(3).hour.do(main)
    while True: #트리거. 음성인식 값이 들어올때 함수 실행
        ft = open('/home/yuice2/envbomi/output_txt2.txt', 'r')
        trigger = text_file.readline()
        if ("감정인식 해줘" in trigger or "감정인식 해줄래" in trigger): #트리거 조건
            playsound.playsound('emotion_rec_start.mp3') #감정인식을 시작합니다(추가)
            ft.close()
'''
def eat_alart():
    playsound(os.path.join(model_dir,"eat_alart.mp3"))

def joke():
    playsound(os.path.join(model_dir,"joke_{0}.mp3".format(joke_time)))

def start():  
    playsound(os.path.join(model_dir,'emotion_rec_start.mp3')) #감정인식을 시작합니다(추가)
    print("session start")

    print("facial_detector start")
    facial_detector()
    print("facial_detector complete")
    
    print("word_inference start")
    word_inference()
    print("word_inference complete")

    print("making emotion statistics")
    emotion_data()
    print("all session complete")
                 
def medicine_alart():
    playsound(os.path.join(model_dir,"medicine.mp3"))
    
def positive_state():
    playsound(os.path.join(model_dir,"normal_{0}.mp3".format(joke_time %4)))

def negative_state():
    playsound(os.path.join(model_dir,"negative_{0}.mp3".format(joke_time %4)))

def normal_state():
    playsound(os.path.join(model_dir,"normal_{0}.mp3".format(joke_time %4)))


def main():
    global emotion_tier
    print("main start")
    playsound(os.path.join(model_dir,"start_voice.mp3"))
    print("1")
    if (tm.tm_mon == "10" and tm.tm_mday == "3"):
        playsound(os.path.join(model_dir,"bomi_birthday.mp3"))
    schedule.every(3).hours.do(start)
    schedule.every().day.at("{0}:{1}".format(meal_time_breakfast_hour,meal_time_min)).do(eat_alart) #7시 랜덤분 식사 하셨나요?
    schedule.every().day.at("{0}:{1}".format(meal_time_launch_hour,meal_time_min)).do(eat_alart) #12시 랜덤분 
    schedule.every().day.at("{0}:{1}".format(meal_time_dinner_hour ,meal_time_min)).do(eat_alart) #18시 랜덤분
    schedule.every().day.at("08:00").do(medicine_alart) #8시 30분 약 알림
    schedule.every().day.at("13:00").do(medicine_alart) #13시 약 알림
    schedule.every().day.at("19:00").do(medicine_alart) #19시 약 알림
    schedule.every().day.at("{0}:{1}".format(joke_time[0],  rand_min)).do(joke)
    schedule.every().day.at("{0}:{1}".format(joke_time[1],  rand_min)).do(joke)     
    schedule.every().day.at("{0}:{1}".format(joke_time[2],  rand_min)).do(joke)
    
    while True: #트리거. 음성인식 값이 들어올때 함수 실행
        
        schedule.run_pending()
        time.sleep(2)
        ft = open('/home/yuice2/envbomi/output_txt2.txt', 'r') #음성인식 
        trigger = ft.readline() 
        ft.close()
        print(trigger)
        if ("감정 인식" in trigger or "감정인식" in trigger): #트리거 조건 ("감정인식"이 인식된다면)
            print("정상")
            start()
        if ("약 먹었어" in trigger or "약먹었어" in trigger): #약 복용
            playsound(os.path.join(model_dir,"medicine_eat.mp3"))
            fm = open(os.path.join(model_dir,'medicine_eat.txt'), 'w') #쓰기모드로 염 나중에 어팬드 모드로 바꿔서 웹 연동
            medicine_data = "약 복용 완료했습니다. {0}년 {1}월, {2}일 {3}시 {4}분 \n".format(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min)
            fm.write(medicine_data)
            fm.close()
        if ("약먹었니" in trigger or "약 먹었니" in trigger): #약 복용 확인
            fm = open(os.path.join(model_dir,'medicine_eat.txt'), 'r')
            read_data = fm.readline()
            print(read_data)
            tts = gTTS(read_data, lang = 'ko', slow=False)
            time.sleep(2)
            tts.save('medicine_check.mp3')
            time.sleep(2)
            playsound(os.path.join(model_dir,"medicine_check.mp3"))
            fm.close()
        
        #if (emotion_tier == 1):
            #schedule.every().day.at("{0}:{1}".format(rand_num,  rand_min)).do(positive_state)
            #positive_state()
        #elif (emotion_tier == 0):
            #schedule.every().day.at("{0}:{1}".format(rand_num,  rand_min)).do(normal_state)
            #normal_state()
        #elif (emotion_tier == -1):
            #schedule.every().day.at("{0}:{1}".format(rand_num,  rand_min)).do(negative_state)                         
            #negative_state
if __name__ == "__main__" :
    main()
