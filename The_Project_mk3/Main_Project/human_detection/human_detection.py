"""
Project: Human_detection
작성자: 김민관
프로그램 내용
 - 이 프로그램은 카메라를 이용하여 사물을 감지하는 프로그램이다.
 - 모든 사물을 감지하고, 사물이 person인 것을 찾는다.
 - 이 프로그램은 텐서 모듈과 openCV를 기반으로 제작되었다.
 - 텐서 모델은 moblienet_ssd_v2_coco 기계학슴 모듈을 사용한다.
 - 프로그램은 jiteshsaini가 진행한 AI robot프로젝트 도중에 사용된 object_detection을
 기반으로 작성되었으며, 원래 프로그램에서 하드웨어 가속과, 필요없는 부분들을 제거하고 최적화하였다.
"""

import common1 as cm
import cv2
import numpy as np
from PIL import Image
import time

import sys
sys.path.insert(0, './')

cap = cv2.VideoCapture(0)
threshold=0.2 #??? 이거 왜씀?
top_k=5 #감지되어 보여줄수 있는 오브젝트의 갯수

model_dir = './'  #폴더 위치
model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite' #mobilenet_ssd_v2_coco 기계학습 모듈 일반 버번
lbl = 'coco_labels.txt' #라벨(오브젝트 이름이 있는 폴더)

counter=0 #카운터 변수 초기화
prev_val=0 #

selected_obj="person" #따로 지정하지 않고 그냥 사람을 찾도록 함
    
def show_selected_object_counter(objs,labels): #발견한 오브젝트의 라벨을 찾는다. (카메라화면과 텐서 데이터를 연산하여 나온 데이터, 텐서에서 가지고온 라벨)
    global counter, prev_val, selected_obj
    
    arr=[]
    for obj in objs:
        #print(obj.id)
        label = labels.get(obj.id, obj.id) #튜플 서브 클래스에 저장되어 있는 id를 불러옴(라벨 id)
        #print(label)
        arr.append(label)
            
    print("arr:",arr)
    
    x = arr.count(selected_obj) #선택된 오브젝트가 등장하는 횟수

    '''
    if(x>0): 사람이 화면에 있다면 x값이 0보다 큼
        LED를 켜거나 사람이 있다고 표현할 것
    else:
        LED를 끄거나 사람이 없다고 표현할 것
    '''

    diff=x - prev_val #이전 프레임에 관하여 선택된 오브젝트의 발생 횟수의 변화를 diff에 저장
    
    print("diff:",diff) #변화를 출력

    if(diff>0): #오브젝트 발생횟수에 변화가 있다며 카운터 증가
        counter=counter + diff
        
    prev_val = x #전에 오브젝트가 등장했던 횟수를 저장
    
    print("counter:",counter) #카운터를 출력
    
def main():
    mdl = model  #가속없이 라즈베리파이에서 연산하면 속도는 130 ~ 160ms
        
    interpreter, labels =cm.load_model(model_dir,mdl,lbl) #텐서의 라벨과 인터프리터를 가지고옴
    
    fps=1  #fps 변수 초기화
    arr_dur=[0,0,0]  #시간 측정후 기록용 배열 초기화

    while True:
        start_time=time.time() #while문 시작 시간 체크
        
        #----------------Capture Camera Frame-----------------
        start_t0=time.time() #카메라 캡쳐 시간 체크
        ret, frame = cap.read() #비디오의 한프레임씩 읽음, 제대로 프레임을 읽으면 ret값이  True, 실패하면  False가 됨 frame에 읽은 프레임을 기록
        if not ret:  #만약 제대로 읽지 못했다면 ret이 False가 되고 그대로 while문 탈출
            print("someting wrong")
            break
        
        cv2_im = frame  #상하좌우 반전을 통해 카메라에 비치는 비디오를 바르게 배치함(필요없으면 제거)
        cv2_im = cv2.flip(cv2_im, 0) #프레임의 상하 반전
        cv2_im = cv2.flip(cv2_im, 1) #프레임의 좌우 반전

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB) #openCV에서는 BGR순으로 저장되기때문에 RGB순으로 재정렬
        pil_im = Image.fromarray(cv2_im_rgb) #NumPy 배열로 이루어진 이미지 배열을 PIL이미지로 변경 (PIL = Python Image Library)
       
        arr_dur[0]=time.time() - start_t0 #캡쳐하는데 걸리는 시간 계산
        cm.time_elapsed(start_t0,"camera capture")
        #----------------------------------------------------
       
        #-------------------Inference---------------------------------
        start_t1=time.time() #추론하는데 걸리는 시간 체크

        cm.set_input(interpreter, pil_im) #가지고온 텐서 인터프리터 모델과 이미지를 인풋함
        interpreter.invoke()  #텐서 인터프리터의 연산 권한을 위임함
        objs = cm.get_output(interpreter, score_threshold=threshold, top_k=top_k) #인터프리터 모델과 이미지를 대조하여 분석한 결과를 objs에 저장
        
        arr_dur[1]=time.time() - start_t1  #추론하는데 걸린 시간 측정
        cm.time_elapsed(start_t1,"inference")
        #----------------------------------------------------
       
       #-----------------other------------------------------------
       #오브젝트를 찾고 스트리밍 하는 부분
        start_t2=time.time() #시작 시간 체크

        show_selected_object_counter(objs,labels)#오브젝트를 찾는 핵심 부분 함수
        

        if cv2.waitKey(1) & 0xFF == ord('q'): #waitkey로 24비트 입력값을 받아서 oxFF로 비트마스킹을 하여 32비트 ord('q')와 같은지 비교 
            break #즉, q를 입력하면 반복문 종료
        
        
        cv2_im = cm.append_text_img1(cv2_im, objs, labels, arr_dur, counter,selected_obj)#사각형으로 오브젝트를 감싸게 하는 함수
        cv2.imshow('Object Detection - TensorFlow Lite', cv2_im) #미리보기
       
        arr_dur[2]=time.time() - start_t2 #오브젝트를 찾아서 스트리밍하는데 까지 걸린 시간 측정
        cm.time_elapsed(start_t2,"other")
        cm.time_elapsed(start_time,"overall")
        
        print("arr_dur:",arr_dur)
        fps = round(1.0 / (time.time() - start_time),1) #총 사용시간 측정후 fps단위로 연산 후 출력
        print("*********FPS: ",fps,"************")

    cap.release()  #사용한 영상 리소스 반환
    cv2.destroyAllWindows() #cv2로 인해 열린 모든 윈도우 창을 닫음


if __name__ == '__main__':
    main()