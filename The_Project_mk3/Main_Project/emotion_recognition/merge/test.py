import multiprocessing
from playsound import playsound

play = multiprocessing.Process(target=playsound, args = ("AmorParty.mp3",))

try:
    play.start()
except KeyboardInterrupt:
    print("1")


