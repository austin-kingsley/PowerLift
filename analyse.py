import numpy as np
import matplotlib.pyplot as plt

def getAverageRepDuration(repTimes, fps, multiplier):
    # count number of frames between start and end of rep, get mean across all reps, then divide by framerate
    return np.mean([r[2] - r[0] for r in repTimes])/fps


def getAverageConcentricRepDuration(repTimes, fps, isDeadlift, multiplier):
    # count number of frames between start and end of the concentric portion of the rep, get mean across all reps, then divide by framerate
    if isDeadlift:
        return np.mean([r[1] - r[0] for r in repTimes])/fps
    return np.mean([r[2] - r[1] for r in repTimes])/fps


def getPeakConcentricVelocity(repTimes, v, isDeadlift, multiplier):
    l = []
    # iterate over each rep and find the maximum velocity within the concentric portion
    for r in repTimes:
        l.append(max(v[r[0]:r[1]]) if isDeadlift else max(v[r[1]:r[2]]))
    return max(l) # return the maximum maximum across all reps


    # if isDeadlift:
    #     # iterate over each rep and find the maximum velocity within the concentric portion
    #     for r in repTimes:
    #         l.append(max(v[r[0]:r[1]]))
    #     return max(l) # return the maximum maximum across all reps
    # else:
    #     # iterate over each rep and find the maximum velocity within the concentric portion
    #     for r in repTimes:
    #         l.append(max(v[r[1]:r[2]]))
    #     return max(l) # return the maximum maximum across all reps

def getAverageConcentricVelocity(repTimes, v, isDeadlift, multiplier, fps, rom, foundPlate):
    # a = abs(rom/getAverageConcentricRepDuration(repTimes, fps, isDeadlift, multiplier)) if foundPlate else abs(1/getAverageConcentricRepDuration(repTimes, fps, isDeadlift, multiplier))
    l = []
    # iterate over each rep and find the mean velocity within the concentric portion
    for r in repTimes:
        l.append(np.mean(v[r[0]:r[1]]) if isDeadlift else np.mean(v[r[1]:r[2]]))
    return np.mean(l) # return the average mean across all reps
    
    
    # l = []
    # if isDeadlift:
    #     for r in repTimes:
    #         l.append(np.mean(v[r[0]:r[1]]))
    #     return np.mean(l)
    # else:
    #     for r in repTimes:
    #         l.append(np.mean(v[r[1]:r[2]]))
    #     return np.mean(l)
    

    
def getStickingPoint(repTimes, v, d, isDeadlift, rom):
    bottomMeanMean, topMeanMean = 0, 0

    # loop through each rep
    for r in repTimes:
        # extract concentric velocity and displacement
        concentricV = v[r[0]:r[1]] if isDeadlift else v[r[1]:r[2]]
        concentricD = d[r[0]:r[1]] if isDeadlift else d[r[1]:r[2]]

        # separate concentric velocity into bottom and top half of ROM using concentric displacement
        bottomHalfVeloctiy = [concentricV[i] for i in range(len(concentricV)) if concentricD[i] < rom/2]
        topHalfVeloctiy = [concentricV[i] for i in range(len(concentricV)) if concentricD[i] > rom/2]

        # find average velocity for bottom and top halves and print.
        bottomMean = np.average(bottomHalfVeloctiy)
        topMean = np.average(topHalfVeloctiy)
        print("Bottom mean: " + str(bottomMean))
        print("Top mean: " + str(topMean))
        print()
        # accumulate to get mean average bottom and top velocities
        bottomMeanMean += bottomMean/len(repTimes)
        topMeanMean += topMean/len(repTimes)

    print("Overall bottom mean: " + str(bottomMeanMean))
    print("Overall top mean: " + str(topMeanMean))

    # If the ratio between bottom and top is within 1.1x, return no sticking point
    if max(bottomMeanMean, topMeanMean)/min(bottomMeanMean, topMeanMean) < 1.1:
        return "None"
    return "Bottom" if bottomMean < topMean else "Top" # else sticking point is slowest of the two regions

def getAnalysis(startY, endY, repTimes, fpsList, framerate, bbox, isDeadlift, v, d, foundPlate, t, filename):

    # error check for no reps
    if not repTimes:
        print("No reps recorded")
        return

    # error check for malformed repTimes list
    if type(repTimes[-1]) != tuple or len(repTimes[-1]) != 3:
        repTimes = repTimes[:-1]

    # error check for no reps once more
    if not repTimes:
        print("No full reps recorded")
        return

    pixelsPerRep = startY - endY
    avgFPS = np.mean(fpsList)
    multiplier = 1

    # if weight plate is detected, convert v and d to metres/second and metres respectively
    if foundPlate:
        pixelsPerWeightPlate = abs(bbox[3])
        metersPerWeightPlate = 0.45
        metersPerPixel = metersPerWeightPlate/pixelsPerWeightPlate
        rom = pixelsPerRep*metersPerPixel
        v = [metersPerWeightPlate*x/pixelsPerWeightPlate for x in v]
        d = [metersPerWeightPlate*(startY - x)/pixelsPerWeightPlate for x in d]
    else: # else, convert v and d to reps/second and reps respectively
        pixelsPerRep = abs(startY - endY)
        v = [x/pixelsPerRep for x in v]
        d = [(startY - x)/pixelsPerRep for x in d]
        rom = pixelsPerRep

    # calculate performance metrics
    averageRepDuration = getAverageRepDuration(repTimes, avgFPS, multiplier)
    averageConcentricRepDuration = getAverageConcentricRepDuration(repTimes, avgFPS, isDeadlift, multiplier)
    peakConcentricVelocity = getPeakConcentricVelocity(repTimes, v, isDeadlift, multiplier)
    averageConcentricVelocity = getAverageConcentricVelocity(repTimes, v, isDeadlift, multiplier, avgFPS, rom, foundPlate)
    stickingPoint = getStickingPoint(repTimes, v, d, isDeadlift, rom)

    print()  # print performance metrics
    print("Average rep duration:               " + str(round(averageRepDuration, 3)) + " seconds")
    print("Average concentric rep duration:    " + str(round(averageConcentricRepDuration, 3)) + " seconds")
    print("Peak concentric velocity:           " + str(round(peakConcentricVelocity, 3)) + (" metres/second" if foundPlate else " reps/second"))
    print("Average concentric velocity:        " + str(round(averageConcentricVelocity, 3)) + (" metres/second" if foundPlate else " reps/second"))
    print("Sticking point:                     " + stickingPoint)

    # let x axis be time in seconds, as opposed to frame count (t)
    xAxis = [x/avgFPS for x in range(t)]

    # trim x axis to start of first rep and end of last rep
    startTime = repTimes[0][0]
    endTime = repTimes[-1][2]
    v = v[startTime:endTime]
    d = d[startTime:endTime]
    xAxis = xAxis[startTime:endTime]

    # plot velocity and displacement over time
    plt.plot(xAxis, v, label="velocity")
    plt.plot(xAxis, d, label="displacement")
    plt.xlabel("time (seconds)")
    plt.ylabel("height (metres)")
    plt.legend(loc="lower left" if isDeadlift else "upper left")
    plt.savefig(filename + ".png")
    plt.savefig("foo.png")
    plt.clf()
    plt.cla()

    # return performance metrics rounded to 3 decimal places
    return (str(round(averageRepDuration, 3)) + " seconds", str(round(averageConcentricRepDuration, 3)) + " seconds", str(round(peakConcentricVelocity, 3)) + (" metres/second" if foundPlate else " reps/second"), str(round(averageConcentricVelocity, 3)) + (" metres/second" if foundPlate else " reps/second"), stickingPoint)