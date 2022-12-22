import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

forword = 17
backword = 27
left = 22
right = 23
delay = 1
def pre():
    GPIO.setup(17,GPIO.OUT)
    GPIO.setup(27,GPIO.OUT)
    GPIO.setup(22,GPIO.OUT)
    GPIO.setup(23,GPIO.OUT)

def forword():
    GPIO.output(17,True)
    time.sleep(delay)

def backword():
    GPIO.output(23,True)
    time.sleep(delay)

def right():
    GPIO.output(27,True)
    time.sleep(delay)

def left():
    GPIO.output(22,True)
    time.sleep(delay)

def init():
    GPIO.output(22,False)
    GPIO.output(17,False)
    GPIO.output(27,False)
    GPIO.output(23,False)
