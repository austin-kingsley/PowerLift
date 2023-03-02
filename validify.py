def goodDepth(img, detector):
    lm = detector.findPosition(img, False)

    if not lm:
        return "cant find lifter"

    mostVisibleKnee = 25 if lm[25][3] > lm[26][3] else 26

    kneeJoint = lm[mostVisibleKnee]

    hipJoint = lm[mostVisibleKnee-2]

    return hipJoint[1] > kneeJoint[1]


def lockedOut(img, detector):
    lm = detector.findPosition(img, False)

    mostVisibleShoulder = 12 if lm[12][3] > lm[11][3] else 11

    lookingRight = lm[mostVisibleShoulder+20][0] > lm[mostVisibleShoulder+12][0]

    # foot, hip, shoulder
    a = detector.findAngle(img, mostVisibleShoulder+16, mostVisibleShoulder+12, mostVisibleShoulder, draw=False)

    a = 360 - a if lookingRight else a

    return a >= 170