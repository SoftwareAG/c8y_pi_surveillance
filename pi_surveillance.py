# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2
import numpy as np
import requests

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="path to the JSON configuration file")
args = vars(ap.parse_args())
conf = json.load(open(args["conf"]))
auth = (conf['c8y_username'],conf['c8y_password']) 
url = conf['c8y_baseurl']
f = open('/proc/cpuinfo','r')
for line in f:
	if line[0:6]=='Serial':
		cpuserial = line[10:26]
f.close()
r = requests.get(url + '/identity/externalIds/c8y_Serial/' + cpuserial, auth=auth, headers={'Accept':'application/json'})
if r.status_code == 200:
	device_id = r.json()['managedObject']['id']
else:
	device_id = requests.post(url + '/inventory/managedObjects', auth=auth, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}, json={'name': 'PI ' + cpuserial})
	requests.post(url + '/identity/globalIds/' + device_id + '/externalIds', auth=auth, json={'type': 'c8y_Serial', 'externalId': cpuserial})
print("Device Id is {}".format(device_id))
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("mobilenet.prototxt", "mobilenet.caffemodel")

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))
# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0

# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image and initialize
	# the timestamp and occupied/unoccupied text
	frame = f.array
	(H, W) = frame.shape[:2]
	timestamp = datetime.datetime.now()
	text = "Unoccupied"
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	# if the average frame is None, initialize it
	if avg is None:
		print("[INFO] starting background model...")
		avg = gray.copy().astype("float")
		rawCapture.truncate(0)
		continue
	# accumulate the weighted average between the current frame and
	# previous frames, then compute the difference between the current
	# frame and running average
	cv2.accumulateWeighted(gray, avg, 0.5)
	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
	# threshold the delta image, dilate the thresholded image to fill
	# in holes, then find contours on thresholded image
	thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
		cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < conf["min_area"]:
			continue
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		#cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Occupied"
	# draw the text and timestamp on the frame
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (0, 0, 255), 1)
	# check to see if the room is occupied
	if text == "Occupied":
		# check to see if enough time has passed between uploads
		if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
			# increment the motion counter
			motionCounter += 1
			# check to see if the number of frames with consistent motion is
			# high enough
			if motionCounter >= conf["min_motion_frames"]:
				# send event to Cumulocity
				persons = 0
				img = frame
				(h, w, c) = frame.shape
				if w>h:
					dx = int((w-h)/2)
					img = frame[0:h, dx:dx+h]
				resized = cv2.resize(img, (300, 300), cv2.INTER_AREA)
				blob = cv2.dnn.blobFromImage(resized, 1.0/127.5, (300, 300), 127.5)
				net.setInput(blob)
				detections = net.forward()
				persons = 0
				# loop over the detections
				for i in np.arange(0, detections.shape[2]):
					# extract the confidence (i.e., probability) associated with the
					# prediction
					confidence = detections[0, 0, i, 2]
					# filter out weak detections by ensuring the `confidence` is
					# greater than the minimum confidence
					if confidence > 0.95:
						# extract the index of the class label from the `detections`,
						# then compute the (x, y)-coordinates of the bounding box for
						# the object
						idx = int(detections[0, 0, i, 1])
						print("Detected {}".format(CLASSES[idx]))
						box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box.astype("int")
						if CLASSES[idx] == 'person':
							persons = persons + 1
						# display the prediction
						label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
						print("[INFO] {}".format(label))
						cv2.rectangle(frame, (startX, startY), (endX, endY),
							COLORS[idx], 2)
						y = startY - 15 if startY - 15 > 15 else startY + 15
						cv2.putText(frame, label, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
				print("{} persons detected".format(persons))
				# write the image to temporary file
				event_data = {'source': {'id': device_id}, "persons":persons, "text": "Room is occupied", "type": "room_occupied", "time": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()}
				event = requests.post(url + '/event/events',  auth=auth, headers={"Content-Type": "application/json", "Accept": "application/json"}, json=event_data ).json()
				requests.post(url + '/event/events/' + event['id'] + '/binaries', auth=auth, headers={"Accept": "application/json"}, files={'file': ('snapshot.jpg', cv2.imencode('.jpg', frame)[1].tobytes(), 'image/jpg')})
				# update the last uploaded timestamp and reset the motion
				# counter
				lastUploaded = timestamp
				motionCounter = 0
	# otherwise, the room is not occupied
	else:
		motionCounter = 0
	# check to see if the frames should be displayed to screen
	if conf["show_video"]:
		# display the security feed
		cv2.imshow("Security Feed", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key is pressed, break from the lop
		if key == ord("q"):
			break
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
