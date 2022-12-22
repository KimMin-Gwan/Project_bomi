
from flask import Flask,request,jsonify

app = Flask(__name__)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(17,GPIO.OUT)
GPIO.output(17,0)

@app.route('/keyboard')
def keyboard():
	dataSend = {
		"type" : "text"
		}
		
	return jsonify(dataSend)
	
@app.route('/message',methods=['post'])
def message():
	
	dataReceive = request.get_json()
	content = dataReceive['content']
	
	if content == "led on":
		dataSend = {
			"message" : {
				"text" : "led on"
			}
		}
		GPIO.output(17,1)
	elif content == "led off":
		dataSend = {
			"message" : {
				"text" : "led off"
			}
		}
		GPIO.output(17,0)
	else:
		dataSend = {
			"message" : {
				"text" : "'led on' or 'led off'"
			}
		}
		
	return jsonify(dataSend)
	
if __name__ == "__main__":
	app.run(host="0.0.0.0",port = 5050, debug=True)