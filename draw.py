import cv2

# trace around a given bounding box
def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x,y), ((x+w),(y+h)), (255,0,255), 3, 1)
    return img

# trace the bar path
def drawPath(img, bbox, smoothedVelocity, imgCanvas, xp, yp):
    # find centre of bar
    centreX = int(bbox[0] + bbox[2]/2)
    centreY = int(bbox[1] + bbox[3]/2)

    # if this is the first frame, set the previous point to the current point
    if xp == 0 and yp == 0:
        xp, yp = centreX, centreY

    # find velocity in metres per second
    pixelsPerWeightPlate = abs(bbox[3])
    metersPerWeightPlate = 0.45 # 0.442 m
    v = abs(metersPerWeightPlate * smoothedVelocity /pixelsPerWeightPlate) if pixelsPerWeightPlate else 0

    # define colour gradient
    green = (0, 255, 0)
    orange = (0, 165, 255)
    red = (0, 0, 255)
    gradient = [red, orange, green]

    # determine color based on velocity
    velocity_range = (0, 0.5) # m/s
    velocity_norm = (v - velocity_range[0]) / (velocity_range[1] - velocity_range[0])
    color_norm = (len(gradient) - 1) * velocity_norm
    color_index = int(color_norm)
    color_remainder = color_norm - color_index
    colour = green if v >= velocity_range[1] else tuple(int((1 - color_remainder) * gradient[color_index][i] + color_remainder * gradient[color_index + 1][i]) for i in range(3))

    
    # determine thickness based on the width of bounding box
    thickness = int(bbox[2]/40)
    thickness = thickness if thickness else 5

    # draw line on image canvas
    cv2.line(imgCanvas, (xp, yp), (centreX, centreY), colour, thickness)
    
    # mask bar path from image canvas onto original image
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # return modified image and current point
    return img, centreX, centreY