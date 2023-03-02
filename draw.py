import cv2

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x,y), ((x+w),(y+h)), (255,0,255), 3, 1)
    return img

def drawPath(img, bbox, smoothedVelocity, topSpeed, imgCanvas, xp, yp):
    centreX = int(bbox[0] + bbox[2]/2)
    centreY = int(bbox[1] + bbox[3]/2)

    if xp == 0 and yp == 0:
        xp, yp = centreX, centreY

    colourSpeed = 510 * abs(smoothedVelocity)/(topSpeed + 1)

    # colour = (0, 255-min(colourSpeed, 255), min(colourSpeed , 255))
    colour = (0, min(colourSpeed , 255), 255-min(colourSpeed, 255))

    cv2.line(imgCanvas, (xp, yp), (centreX, centreY), colour, 5)

    # xp, yp = centreX, centreY

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    return img, centreX, centreY