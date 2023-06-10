# check if squat is valid
def goodDepth(img, detector):
    # extract landmarks
    lm = detector.findPosition(img, False)
    if not lm: 
        return False

    # identify the most visible knee, get corresponding hip
    mostVisibleKnee = 25 if lm[25][3] > lm[26][3] else 26
    hipJoint = mostVisibleKnee - 2

    # length of femur = cartesian distance between knee and hip
    legLength = detector.findDistance(mostVisibleKnee, hipJoint)

    # compare offsetted hip and knee y-values
    return lm[hipJoint][1] > lm[mostVisibleKnee][1] + legLength/30

# check if deadlift is valid
def lockedOut(img, detector):
    # extract landmarks
    lm = detector.findPosition(img, False)
    if not lm:
        return

    # identify the most visible shoulder
    mostVisibleShoulderID = 12 if lm[12][3] > lm[11][3] else 11

    # compare x-value of the foot-tip to hip to determine direction lifter is facing
    lookingRight = lm[mostVisibleShoulderID+20][0] > lm[mostVisibleShoulderID+12][0]

    # find angle made between ankle, hip, and shoulder
    a = detector.findAngle(img, mostVisibleShoulderID+16, mostVisibleShoulderID+12, mostVisibleShoulderID, draw=False)

    # normalize
    a = 360 - a if lookingRight else a

    # compare against threshold
    return a >= 176

# check if bench press is valid
def goodPause(t, pauseStart, framerate):
    numOfPausedFrames = t - pauseStart # get number of paused frames
    pausedTime = numOfPausedFrames/framerate # convert to seconds
    return pausedTime >= 0.62 # compare against threshold
