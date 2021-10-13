#import needed packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import matplotlib.pyplot as plt
import os
import glob

photosTake = 3
def calibrate( frame, calCount):
	
	

	calFrame = frame
	

	
	#os.remove(file) for file in os.listdir('./Pictures/Camera Roll/') if file.endswith('.jpg')
	if (calCount < photosTake ):
		
		cv2.imshow("Frame", calFrame)
		key = cv2.waitKey(30) & 0xFF
		if (key == ord("t")):
			
			print ("t is pressed")
			calCount = calCount + 1
			
			
			cv2.imwrite("test" + str(calCount) + ".jpg", calFrame)
			
	
	# Defining the dimensions of checkerboard
	if (calCount >= photosTake ):
		print ("calibrating")
		calCount = calCount + 1
		
		CHECKERBOARD = (6,8)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

		# Creating vector to store vectors of 3D points for each checkerboard image
		objpoints = []
		# Creating vector to store vectors of 2D points for each checkerboard image
		imgpoints = [] 


		# Defining the world coordinates for 3D points
		objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
		objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
		prev_img_shape = None

		# Extracting path of individual image stored in a given directory
		images = glob.glob('./Pictures/Camera Roll/*.jpg')
		for fname in images:
			img = cv2.imread(fname)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			# Find the chess board corners
			# If desired number of corners are found in the image then ret = true
			ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
			
			"""
			If desired number of corner are detected,
			we refine the pixel coordinates and display 
			them on the images of checker board
			"""
			if ret == True:
				objpoints.append(objp)
				# refining pixel coordinates for given 2d points.
				corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
				
				imgpoints.append(corners2)

				# Draw and display the corners
				img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
			
			cv2.imshow('img',img)
			cv2.waitKey(0)
		
		cv2.destroyAllWindows()

		h,w = img.shape[:2]

		"""
		Performing camera calibration by 
		passing the value of known 3D points (objpoints)
		and corresponding pixel coordinates of the 
		detected corners (imgpoints)
		"""
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

		print("Camera matrix : \n")
		print(mtx)
		print("dist : \n")
		print(dist)
		print("rvecs : \n")
		print(rvecs)
		print("tvecs : \n")
		print(tvecs)
	return calCount 

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
calCount = 0

tennisDiameterM  = 0.0654
tennisRadiusM = 0.0327
maxFactor = 57.23
xDist = 0
xCount = 0
xPos = 0
initialX = 0
travelTime = 2
yDist = 0
yCount = 0
yPos = 0
g = 9.82
velocitySum =0


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
tempXVals = np.empty(10, np.double)
tempYVals = np.empty(100, np.double)
tempTimeVal  =np.empty(len(tempXVals)-1, np.double)
tempVelocity = np.empty(len(tempXVals)-1, np.double)
XDiff = np.empty(len(tempXVals)-1, np.double)
YDiff = np.empty(len(tempYVals)-1, np.double)
avgXCount = 0
xSum = 0
timeSum = 0
timeTwo = time.time()
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

	if (calCount < photosTake+1):
		calCount = calibrate(frame, calCount)

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
				
				if (xCount < len(tempXVals) and xCount >= 0):
					
					if (xCount % 2 == 0):
						initialX = x
						timeOne = time.time()
						if(xCount != len(tempXVals)-1):
							tempTimeVal[xCount] =  timeOne-timeTwo
							
					if (xCount % 2 == 1):
						timeTwo = time.time()
						if(xCount != len(tempXVals)-1):
							tempTimeVal[xCount] = timeTwo - timeOne
							
					tempXVals[xCount] = x
					tempYVals[xCount] = -y +800
					xCount += 1
					#print (str(x))
				
				if (xCount >= len(tempXVals)):
					
					
					
					timeDiff = timeTwo- timeOne
					for j in range (0,len(XDiff)):
						XDiff[j] = (tempXVals[j+1]-tempXVals[j])
						#print (str(XDiff))
						xSum += XDiff[j]
					tempXVals.fill(0)
					for k in range (0,len(tempTimeVal)):
						tempVelocity[k] = XDiff[k]/tempTimeVal[k]
						timeSum += tempTimeVal[k]
						#print (timeSum)
					for h in range (0, len(tempVelocity)):
						velocitySum += tempVelocity[h]
					
					#print (str(timeDiff))
					
					#matplot
					#print (tempXVals)
					#time.sleep(10)
					#plt.plot(tempVelocity, tempTimeVal, color='g', label='xvals')
					#plt.plot(tempXVals, tempYVals , color='g', label='xvals')
					#plt.xlabel('t')
					#plt.ylabel('x')
					#plt.plot(tempXVals, color='r', label = 'xvelocity')
					#plt.legend()
					#plt.show()  
					#print (timeSum/len(tempTimeVal))
					xVelocity = (((xSum / len(XDiff) )*tennisRadiusM/radius)/(timeSum/len(tempTimeVal)))
					#xVelocity = (velocitySum / len(tempVelocity)) 
					
					print ("X Velocity is " + str(xVelocity) + "m/s")
					xCount = 0
					timeSum = 0
					xSum = 0
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


