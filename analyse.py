import numpy as np
import matplotlib.pyplot as plt

def getAverageRepDuration(repTimes, fps, multiplier):
    return multiplier * np.mean([r[2] - r[0] for r in repTimes])/fps


def getAverageConcentricRepDuration(repTimes, fps, isDeadlift, multiplier):
    if isDeadlift:
        return multiplier * np.mean([r[1] - r[0] for r in repTimes])/fps
    return multiplier * np.mean([r[2] - r[1] for r in repTimes])/fps


def getPeakConcentricVelocity(repTimes, v, isDeadlift, multiplier):
    l = []
    if isDeadlift:
        for r in repTimes:
            l.append(max(v[r[0]:r[1]]))
        return max(l) / multiplier
    else:
        for r in repTimes:
            l.append(max(v[r[1]:r[2]]))
        return max(l) / multiplier

def getMinConcentricVelocity(repTimes, v, isDeadlift, multiplier):
    l = []
    if isDeadlift:
        for r in repTimes:
            l.append(min(v[r[0]:r[1]]))
        return min(l) / multiplier
    else:
        for r in repTimes:
            l.append(min(v[r[1]:r[2]]))
        return min(l) / multiplier

def  getAverageConcentricVelocity(repTimes, fps, isDeadlift, multiplier):
    return 1 / getAverageConcentricRepDuration(repTimes, fps, isDeadlift, multiplier) 


def getStickingPoint(repTimes, v, d, isDeadlift):
    bottomMeanMean, topMeanMean = 0, 0
    for r in repTimes:
        concentricV = v[r[0]:r[1]] if isDeadlift else v[r[1]:r[2]]
        concentricD = d[r[0]:r[1]] if isDeadlift else d[r[1]:r[2]]
        bottomHalfVeloctiy = [concentricV[i] for i in range(len(concentricV)) if concentricD[i] < 0.5]
        topHalfVeloctiy = [concentricV[i] for i in range(len(concentricV)) if concentricD[i] > -0.5]
        bottomMean = np.average(bottomHalfVeloctiy)
        topMean = np.average(topHalfVeloctiy)
        print("Bottom mean: " + str(bottomMean))
        print("Top mean: " + str(topMean))
        print()
        bottomMeanMean += bottomMean/len(repTimes)
        topMeanMean += topMean/len(repTimes)
    print("Overall bottom mean: " + str(bottomMeanMean))
    print("Overall top mean: " + str(topMeanMean))
    if max(bottomMeanMean, topMeanMean)/min(bottomMeanMean, topMeanMean) < 1.1:
        return "none"
    return "bottom" if bottomMean < topMean else "top"

def getAnalysis(startY, endY, repTimes, fpsList, framerate, bbox, isDeadlift, v, d, foundPlate, t):

    if not repTimes:
        print("No reps recorded")
        return

    # print("last:")
    # print(repTimes[-1])

    if type(repTimes[-1]) != tuple or len(repTimes[-1]) != 3:
        repTimes = repTimes[:-1]

    if not repTimes:
        print("No full reps recorded")
        return

    pixelsPerRep = startY - endY

    avgFPS = np.mean(fpsList)

    multiplier = avgFPS/framerate

    if foundPlate:
        pixelsPerWeightPlate = abs(bbox[1] - bbox[3]) # 0.442 m
        metersPerWeightPlate = 0.45
        v = [metersPerWeightPlate*x/pixelsPerWeightPlate for x in v]
        d = [metersPerWeightPlate*(startY - x)/pixelsPerWeightPlate for x in d]
    else:
        pixelsPerRep = abs(startY - endY)
        v = [x/pixelsPerRep for x in v]
        d = [(startY - x)/pixelsPerRep for x in d]

    # try and take fps into account for each individual time period
    averageRepDuration = getAverageRepDuration(repTimes, avgFPS, multiplier)
    averageConcentricRepDuration = getAverageConcentricRepDuration(repTimes, avgFPS, isDeadlift, multiplier)
    peakConcentricVelocity = getPeakConcentricVelocity(repTimes, v, isDeadlift, multiplier)
    # minConcentricVelocity = getMinConcentricVelocity(repTimes, v, isDeadlift, multiplier)
    averageConcentricVelocity = getAverageConcentricVelocity(repTimes, avgFPS, isDeadlift, multiplier)
    stickingPoint = getStickingPoint(repTimes, v, d, isDeadlift)


    print()
    print("Average rep duration:               " + str(round(averageRepDuration, 3)) + " seconds")
    print("Average concentric rep duration:    " + str(round(averageConcentricRepDuration, 3)) + " seconds")
    print("Peak concentric velocity:           " + str(round(peakConcentricVelocity, 3)) + (" metres/second" if foundPlate else " reps/second"))
    # print("Minimum concentric velocity:        " + str(round(minConcentricVelocity, 3)) + (" metres/second" if foundPlate else " reps/second"))
    print("Average concentric velocity:        " + str(round(averageConcentricVelocity, 3)) + (" metres/second" if foundPlate else " reps/second"))
    print("Sticking point:                     " + stickingPoint)

    xAxis = [x/avgFPS for x in range(t)]

    startTime = repTimes[0][0]
    endTime = repTimes[-1][2]

    v = v[startTime:endTime]
    d = d[startTime:endTime]
    xAxis = xAxis[startTime:endTime]

    # plt.plot(fpsList)
    plt.plot(xAxis, v)
    plt.plot(xAxis, d)

    # plt.show()
    plt.savefig('foo.png')

    return (str(round(averageRepDuration, 3)) + " seconds", str(round(averageConcentricRepDuration, 3)) + " seconds", str(round(peakConcentricVelocity, 3)) + (" metres/second" if foundPlate else " reps/second"), str(round(averageConcentricVelocity, 3)) + (" metres/second" if foundPlate else " reps/second"), stickingPoint)