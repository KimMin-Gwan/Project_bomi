"""
Project: AI Robot - Object Detection
Author: Jitesh Saini
Github: https://github.com/jiteshsaini
website: https://helloworld.co.in
- The robot uses PiCamera to capture frames. 
- An object within the frame is detected using Machine Learning moldel & TensorFlow Lite interpreter. 
- Using OpenCV, the frame is overlayed with information such as: color coded bounding boxes, information bar to show FPS, Processing durations and an Object Counter.
- The frame with overlays is streamed over LAN using FLASK. The Flask stream is embedded into a Web GUI which can be accessed at 
"http://192.168.185.45/web". IP '192.168.1.20' should be replaced with your RPi's IP
- You can select an object through Web GUI to generate alarm on a specific object.
- Google Coral USB Accelerator can be used to accelerate the inferencing process.
When Coral USB Accelerator is connected, amend line 14 of util.py as:-
edgetpu = 1 
When Coral USB Accelerator is not connected, amend line 14 of util.py as:-
edgetpu = 0 
"""

import common1 as cm
import cv2
import numpy as np
from PIL import Image
import time

import sys
sys.path.insert(0, './')
#import util as ut
#ut.init_gpio() gpio모듈 불러오기

cap = cv2.VideoCapture(0)
threshold=0.2 #??? 이거 왜씀?
top_k=5 #감지되어 보여줄수 있는 오브젝트의 갯수

model_dir = './'  #폴더 위치
model = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite' #mobilenet_ssd_v2_coco 기계학습 모듈 일반 버번
model_edgetpu = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite' #mobilenet_ssd_v2_coco 기계학습 모듈 가속 버번
lbl = 'coco_labels.txt' #라벨(오브젝트 이름이 있는 폴더)

counter=0 #카운터 변수 초기화
prev_val=0 #

file_path="./web/" #웹 폴더 주소
selected_obj="" #jquery형태로 저장된 데이터를 불러오기 때문에 문자열로 저장할것을 초기화
prev_val_obj="" #위와 동일

#---------Flask----------------------------------------
from flask import Flask, Response
from flask import render_template

app = Flask(__name__) #단일 모율이므로 __name__으로 작성

@app.route('/') #url은 아래에 연결된 index함수와 연결된다. 
def index():
    #return "Default Message"
    return render_template("index2.html") #template폴더 안에서 index2.html을 찾아서 실행시킴

@app.route('/video_feed') #url은 video_feed에 대한 ajax요청이 들어오면 아래에 연결된 video_feed()함수와 연결한다.
def video_feed():
    #global cap
    return Response(main(),
                    mimetype='multipart/x-mixed-replace; boundary=frame') #video_feed함수는 main함수를 반환하며 그의 mimtype은 기본형태이며, 미디어 전송이다
                    #mimetype은 웹 서버에서 화면에 뿌릴 데이터의 형식을 의미하며 , multipart는 기본형태를 의미한다
                    #x-mixed-replace에서 x는 비규격화를 의미하고, replace새롭게 대체됨을 의미한다. 즉, 움직이는 이미지나 무비데이터를 의미한다
                    #boundary는 형태를 의미하며 frame단위를 의미한다
                    
#-------------------------------------------------------------

    
def show_selected_object_counter(objs,labels): #발견한 오브젝트의 라벨을 찾는다. (카메라화면과 텐서 데이터를 연산하여 나온 데이터, 텐서에서 가지고온 라벨)
    global counter, prev_val
    global file_path,selected_obj,prev_val_obj
    
    arr=[]
    for obj in objs:
        #print(obj.id)
        label = labels.get(obj.id, obj.id) #튜플 서브 클래스에 저장되어 있는 id를 불러옴(라벨 id)
        #print(label)
        arr.append(label)
            
    print("arr:",arr)
    
    
    f0 = open(file_path + "/object_cmd.txt", "r+")
    selected_obj = f0.read(20)
    f0.close()
    
    if(selected_obj!=prev_val_obj): #만약 선택된 오브젝트가 전에 등장한 오브젝트가 아니라면 counter를 0으로 초기화
        counter=0
    
    prev_val_obj=selected_obj #한번 선택된오브젝트를 두번 선택하지 않기 위해 따로 저장
    
    
    print("selected_obj: ",selected_obj) #선택된 오브젝트 리스트 출력

    
    x = arr.count(selected_obj) #선택된 오브젝트가 등장하는 횟수
    f1 = open(file_path + "/object_found.txt", "w")
    f1.write(str(x))
    f1.close()

    '''
    if(x>0):#selected object present in frame. Make GPIO pin high
        ut.camera_light("ON") 
    else:#selected object absent in frame. Make GPIO pin Low
        ut.camera_light("OFF")
    '''

    diff=x - prev_val #이전 프레임에 관하여 선택된 오브젝트의 발생 횟수의 변화를 diff에 저장
    
    print("diff:",diff) #변화를 출력

    if(diff>0): #오브젝트 발생횟수에 변화가 있다며 카운터 증가
        counter=counter + diff
        
    prev_val = x #전에 오브젝트가 등장했던 횟수를 저장
    
    print("counter:",counter) #카운터를 출력
    

def main():
    from util import edgetpu
    
    if (edgetpu==1): #만약 Google Coral USB가 연결되어 있다면 연산 가속을 사용
        mdl = model_edgetpu  #가속을 사용하면 연산속도는 15 ~ 17ms
    else:
        mdl = model  #가속없이 라즈베리파이에서 연산하면 속도는 130 ~ 160ms
        
    interpreter, labels =cm.load_model(model_dir,mdl,lbl,edgetpu) #텐서의 라벨과 인터프리터를 가지고옴
    
    fps=1  #fps 변수 초기화
    arr_dur=[0,0,0]  #시간 측정후 기록용 배열 초기화

    #while cap.isOpened():
    while True:
        start_time=time.time() #while문 시작 시간 체크
        
        #----------------Capture Camera Frame-----------------
        start_t0=time.time() #카메라 캡쳐 시간 체크
        ret, frame = cap.read() #비디오의 한프레임씩 읽음, 제대로 프레임을 읽으면 ret값이  True, 실패하면  False가 됨 frame에 읽은 프레임을 기록
        if not ret:  #만약 제대로 읽지 못했다면 ret이 False가 되고 그대로 while문 탈출
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
        #cv2.imshow('Object Detection - TensorFlow Lite', cv2_im)
        
        ret, jpeg = cv2.imencode('.jpg', cv2_im) #스트리밍을 위해 다시 인코딩
        pic = jpeg.tobytes() #스트리밍을 위해 raw data로 변환
        
        cv2.imshow('frame', jpeg)
        
        #Flask 스트리밍
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + pic + b'\r\n\r\n') #성능 이슈로 인해 yield방식으로 스트리밍
       
        arr_dur[2]=time.time() - start_t2 #오브젝트를 찾아서 스트리밍하는데 까지 걸린 시간 측정
        cm.time_elapsed(start_t2,"other")
        cm.time_elapsed(start_time,"overall")
        
        print("arr_dur:",arr_dur)
        fps = round(1.0 / (time.time() - start_time),1) #총 사용시간 측정후 fps단위로 연산 후 출력
        print("*********FPS: ",fps,"************")

    cap.release()  #사용한 영상 리소스 반환
    cv2.destroyAllWindows() #cv2로 인해 열린 모든 윈도우 창을 닫음


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True) # Run FLASK (기존 포트 = 2204) 포트 5000은 오픈 포트
    main()
