"""
file name : dataStatistics.py
details : 데이터 셋의 통계
"""
import numpy as np
import pandas as pd
from configparser import Interpolation
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

DATA_PATH = './DATA/'
print('file size')
for file in os.listdir(DATA_PATH):
    if 'txt' in file:
        print(file.ljust(30) + str(round(os.path.getsize(DATA_PATH + file) / 100000, 2)) + 'MB')

train_data = pd.read_csv(DATA_PATH + 'ratings_train.txt', header = 0, delimiter = '\t', quoting = 3)
train_data.head()

#train파일 안에 id, document, label이 포함되어있다.


#리뷰의 길이들을 확인
print('학습데이터 전체 개수 : {}'.format(len(train_data)))
train_length = train_data['document'].astype(str).apply(len)
train_length.head()

#리뷰 통계 정보---------------------------------------------------------------
print('리뷰 길이 최댓값: {}'.format(np.max(train_length)))
print('리뷰 길이 최솟값: {}'.format(np.min(train_length)))
print('리뷰 길이 평균값: {:.2f}'.format(np.mean(train_length)))
print('리뷰 길이 표준편차: {:.2f}'.format(np.std(train_length)))
print('리뷰 길이 중간값: {}'.format(np.median(train_length)))
print('리뷰 길이 제1사분위: {}'.format(np.percentile(train_length,25)))
print('리뷰 길이 제3사분위: {}'.format(np.percentile(train_length,75)))
#-----------------------------------------------------------------------------

#문자열을 제외한 데이터 제거--------------
train_review = [review for review in train_data['document'] if type(review) is str]
#train_review

#한글 폰트 설정(.ttf파일 다운로드 후 실행)
wordcloud = WordCloud(DATA_PATH+'DalseoDarling.ttf').generate(' '.join(train_review))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

print('긍정 리뷰 갯수 : {}'.format(train_data['label'].value_counts()[1]))
print('부정 리뷰 갯수 : {}'.format(train_data['label'].value_counts()[0]))