"""
자동실행을 위한 start(마스터)파일
"""

import os
import time
#local_path = os.path.dirname(os.path.realpath(__file__))
cmd = "python /home/yuice2/Desktop/The_Project_mk3/Main_Project/emotion_recognition/merge/master.py" + " &"
os.system(cmd)

