#import needed packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

originOffset = 259
#cmScale = .5

pos =-1
scaleFactor = 0

refFactor = 20
calFactor =0
zRef = 30
calibrated = False
proceed = False
z = 0

tennisDiameterM  = 0.0654
tennisRadiusM = 0.0327
maxFactor = 57.23
xDist = 0
xCount = 0
xPos = 0
initialX = 0
travelTime = 2
yDist = 0

color = input ("Input the color of the ball you would like tracked: ")
print (color)
colorList = ["green", 29,86,6 ,64, 255, 255, "tennis", 30,30,30, 80,255,255 ]

for x in range (0, 14):
	
	if colorList[x] == color:
		print("true")
		pos = x


if pos == -1:
	print ("That color ball is not found in the database")
	quit()
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
varLower = (colorList[pos+1],colorList[pos+2],colorList[pos+3])
varUpper = (colorList[pos+4], colorList[pos+5], colorList[pos+6])
tempXVals = np.empty(20, np.double)
tempYVals = np.empty(20, np.double)
avgXDiff = np.empty(len(tempXVals)-1, np.double)
avgYDiff = np.empty(len(tempYVals)-1, np.double)
avgXCount = 0
xSum = 0
#VELOCITY IN M/S
xVelocity = 0
#tennisLower = (0,95,215)
#tennisUpper = (255,255,255)




pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

print("If calibrating, please move the ball out to a distance of 30cm from the camera, then move the f key to confirm calibration")
# keep looping
while True:
    # grab the current frame
	frame = vs.read()

    # handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

    # resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, varLower, varUpper)
	#mask = cv2.inRange(hsv, tennisLower, tennisUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	
    # only proceed if at least one contour was found
	if not calibrated:
		if len(cnts) > 0:
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(30) & 0xFF
		# if the 'f' key is pressed, set the referenceFactor to the scaleFactor at that point
		if key == ord("f"):
			calFactor = radius / tennisRadiusM
			print (calFactor)
			calibrated = True
			center = None
		# if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break
	else :
		if not proceed:
			if key == ord("p"):
				proceed = True
		else:
			if len(cnts) > 0:
				# find the largest contour in the mask, then use
				# it to compute the minimum enclosing circle and
				# centroid
				
				c = max(cnts, key=cv2.contourArea)
				((x, y), radius) = cv2.minEnclosingCircle(c)

				#print (str(x) + "," + str(y))
				
				
				#max x value is 598, min is 0
				#max y value is 448, min is 0
				#max radius is 374.3
				#max scaleFactor is 57.23
				M = cv2.moments(c)
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				
				scaleFactor = radius / tennisRadiusM
				#SCALE FACTOR IS LINEAR???
				#If scale factor is not linear, need to use interp function
				diff = maxFactor - scaleFactor
				#diff never goes to 0
				#print ("Diff is " + str(diff))

				
				#z = ((diff) * tennisDiameterCM)*cmScale
				z = (refFactor/scaleFactor) * zRef
				xDist = (tennisRadiusM/radius) * x
				yDist = (tennisRadiusM/ radius) * y
				if (xCount < len(tempXVals)-1):
					if (xCount == 0):
						initialX = x
						timeOne = time.time()
					if (xCount == 1):
						timeTwo = time.time()
					tempXVals[xCount] = xDist
					xCount += 1
					#print (str(x))
				
				if (xCount >=len(tempXVals)):
					
					timeDiff = timeTwo- timeOne
					for j in range (0, len(avgXDiff)-1):
						avgXDiff[j] = (tempXVals[j+1]-tempXVals[j])
						xSum += avgXDiff[j]
					
					#print (str(timeDiff))
					xVelocity = ((xSum / len(avgXDiff))/timeDiff)
					print ("X Velocity is " + str(xVelocity) + "m/s")
					#xCount =0
				xPos = travelTime * scaleFactor *xVelocity + initialX
				#print ("Projected x is " + str(xPos))
				cv2.circle(frame, (int(xPos), 200), 50,
						(0, 255, 255), 2)
				
				#print ("X position is:" + str(xDist))
				#print ("Z position is :" + str(z)) 
				#print ("Scale factor is :" + str(scaleFactor))
				# only proceed if the radius meets a minimum size
				if radius > 10:
					
					# draw the circle and centroid on the frame,
					# then update the list of tracked points
					cv2.circle(frame, (int(x), int(y)), int(radius),
						(0, 255, 255), 2)
					cv2.circle(frame, center, 5, (0, 0, 255), -1)

		# update the points queue
		pts.appendleft(center) 

		# loop over the set of tracked points
		for i in range(1, len(pts)):
			# if either of the tracked points are None, ignore
			# them
			if pts[i - 1] is None or pts[i] is None:
				continue

			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
			cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

		# show the frame to our screen
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(30) & 0xFF
		
    # if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
  
# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
