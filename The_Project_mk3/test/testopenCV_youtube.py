import cv2
 
########### 카메라 대신 youtube영상으로 대체 ############
import pafy
url = 'https://www.youtube.com/watch?v=u_Q7Dkl7AIk'
video = pafy.new(url)
print('title = ', video.title)
print('video.rating = ', video.rating)
print('video.duration = ', video.duration)
 
best = video.getbest(preftype='mp4')     # 'webm','3gp'
print('best.resolution', best.resolution)
 
cap=cv2.VideoCapture(best.url)
#########################################################
 
#cap = cv2.VideoCapture(0) # 0번 카메라
 
# frame 사이즈
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('frame_size =', frame_size)
 
# 코덱 설정하기
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # ('D', 'I', 'V', 'X')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
 
# 이미지 저장하기 위한 영상 파일 생성
out1 = cv2.VideoWriter('./data/record0.mp4',fourcc, 20.0, frame_size)
out2 = cv2.VideoWriter('./data/record1.mp4',fourcc, 20.0, frame_size,isColor=False)
 
while True:
    retval, frame = cap.read()	# 영상을 한 frame씩 읽어오기
    if not retval:
        break   
        
    out1.write(frame)	# 영상 파일에 저장   
    
    # 이미지 컬러 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    out2.write(gray)	# 영상 파일에 저장        
    
    cv2.imshow('frame',frame)	# 이미지 보여주기
    cv2.imshow('gray',gray)      
    
    key = cv2.waitKey(25)
    if key == 27:
        break
        
cap.release()	# 객체 해제
out1.release()
out2.release()
cv2.destroyAllWindows()