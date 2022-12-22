"""
file name : inference.py
details : txt로 받아온 음성인식 데이터를 긍정과 부정으로 구분
 - 정확도가 65를 넘어갈때만 인정 그외는 모두 의미없는 데이터로 취급
 - emotion_result.txt에 저장 하여 전달

version 1.0  -  inference.py 작성
version 1.1  -  파일 입출력 수정
lastest version : 1.1
작성자 : 김민관, 박세빈
"""

import re
import json
from konlpy.tag import Okt
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
import keras

okt = Okt()
tokenizer  = Tokenizer()

DATA_CONFIGS = 'data_configs.json'
prepro_configs = json.load(open('./CLEAN_DATA/'+DATA_CONFIGS,'r')) #TODO 데이터 경로 설정

#TODO 데이터 경로 설정
with open('./CLEAN_DATA/tokenizer.pickle','rb') as handle:
    word_vocab = pickle.load(handle)

def word_inference():
  
    prepro_configs['vocab'] = word_vocab

    tokenizer.fit_on_texts(word_vocab)

    MAX_LENGTH = 8 #문장최대길이
    fp = open('home/yuice2/envbomi/emotion_input.txt', 'r') #Google Assistance에서 받아올 값 파일로 open
    sentence = fp.readlines()[-1] #sentence에 값 저장, 마지막 줄 값 읽어오기, 정상 작동 확인
    fp.close()

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
        f = open('emotion_reuslt.txt', 'w')
        print(index, file=f)
        f.close()
    else:
        temp = (1 - predictions) * 100
        if(temp < 65):
            index = 0 #nomal result (meanless data)
        else:
            index = -1  #negative result
        f = open('emotion_reuslt.txt', 'w')
        print(index, file=f)
        f.close()
    return 1
