import sys
import os
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci

class SumoIntersection:
    def __init__(self):
        # we need to import python modules from the $SUMO_HOME/tools directory
        try:
            sys.path.append(os.path.join(os.path.dirname(
                __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
            sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
                os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
        except ImportError:
            sys.exit(
                "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

    def getState(self, I, action, tentative_action):
        positionMatrix = []
        velocityMatrix = []

        cellLength = 2.7
        sizeMatrix = 60
        sizeLaneMatric = sizeMatrix / 2 - 12  # 18
        offset = 500 - cellLength * sizeLaneMatric  # 427.1
        offset_out = cellLength * sizeLaneMatric  # 72.9
        speedLimit = 14
        # print(traci.edge.getWaitingTime('gneE21'))

        # junctionPosition = traci.junction.getPosition('0')[0]
        vehicles_road1_in = traci.edge.getLastStepVehicleIDs('gneE21')
        vehicles_road1_out = traci.edge.getLastStepVehicleIDs('gneE22')
        vehicles_road2_in = traci.edge.getLastStepVehicleIDs('gneE86')
        vehicles_road2_out = traci.edge.getLastStepVehicleIDs('gneE87')
        vehicles_road3_in = traci.edge.getLastStepVehicleIDs('gneE89')
        vehicles_road3_out = traci.edge.getLastStepVehicleIDs('gneE88')
        vehicles_road4_in = traci.edge.getLastStepVehicleIDs('gneE85')
        vehicles_road4_out = traci.edge.getLastStepVehicleIDs('gneE84')

        for i in range(sizeMatrix):
            positionMatrix.append([])
            velocityMatrix.append([])
            for j in range(sizeMatrix):
                positionMatrix[i].append(0)
                velocityMatrix[i].append(0)

        index = 42
        offset = 210.62 - cellLength * sizeLaneMatric
        for v in vehicles_road1_in:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = traci.vehicle.getLanePosition(v) - offset
                if std >= 0:
                    temp = std / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (
                            traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][endPos] = 1
                        velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][endPos] = traci.vehicle.getSpeed(v)
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][endPos] = 1
                            velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][
                                endPos] = traci.vehicle.getSpeed(v)
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yEndPos][endPos] = 1
                        velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yEndPos][
                            endPos] = traci.vehicle.getSpeed(v)
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][endPos] = 1
                            velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][
                                endPos] = traci.vehicle.getSpeed(v)
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][endPos] = 1
                            velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][
                                endPos] = traci.vehicle.getSpeed(v)

                    temp = (std - traci.vehicle.getLength(v)) / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0
                        if temp_y_start < 0:
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][startPos] = 1
                            velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][
                                startPos] = traci.vehicle.getSpeed(v)
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][startPos] = 1
                                velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][
                                    startPos] = traci.vehicle.getSpeed(v)
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][startPos] = 1
                            velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][
                                startPos] = traci.vehicle.getSpeed(v)
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][startPos] = 1
                                velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][
                                    startPos] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + j][i] = 1
                                    velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + j][
                                        i] = traci.vehicle.getSpeed(v)
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][startPos] = 1
                                velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][
                                    startPos] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + j][i] = 1
                                    velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + j][
                                        i] = traci.vehicle.getSpeed(v)

        offset = 210.62 - cellLength * sizeLaneMatric
        for v in vehicles_road2_in:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = traci.vehicle.getLanePosition(v) - offset
                if std >= 0:
                    # if traci.vehicle.getVehicleClass(v) == 'taxi':
                    temp = std / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (
                            traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (
                            traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[sizeMatrix - 1 - endPos][
                            index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                        velocityMatrix[sizeMatrix - 1 - endPos][
                            index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = traci.vehicle.getSpeed(v)
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[sizeMatrix - 1 - endPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                            velocityMatrix[sizeMatrix - 1 - endPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = traci.vehicle.getSpeed(v)
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[sizeMatrix - 1 - endPos][index - traci.vehicle.getLaneIndex(v) * 4 + yEndPos] = 1
                        velocityMatrix[sizeMatrix - 1 - endPos][
                            index - traci.vehicle.getLaneIndex(v) * 4 + yEndPos] = traci.vehicle.getSpeed(v)
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[sizeMatrix - 1 - endPos][index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1
                            velocityMatrix[sizeMatrix - 1 - endPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + i] = traci.vehicle.getSpeed(v)
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[sizeMatrix - 1 - endPos][index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1
                            velocityMatrix[sizeMatrix - 1 - endPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + i] = traci.vehicle.getSpeed(v)

                    temp = (std - traci.vehicle.getLength(v)) / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0
                        if temp_y_start < 0:
                            positionMatrix[sizeMatrix - 1 - startPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                            velocityMatrix[sizeMatrix - 1 - startPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = traci.vehicle.getSpeed(v)
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[sizeMatrix - 1 - startPos][
                                    index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                                velocityMatrix[sizeMatrix - 1 - startPos][
                                    index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = traci.vehicle.getSpeed(v)
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[sizeMatrix - 1 - startPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                            velocityMatrix[sizeMatrix - 1 - startPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = traci.vehicle.getSpeed(v)
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[sizeMatrix - 1 - startPos][
                                    index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1
                                velocityMatrix[sizeMatrix - 1 - startPos][
                                    index - traci.vehicle.getLaneIndex(v) * 4 + i] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[sizeMatrix - 1 - i][
                                        index - traci.vehicle.getLaneIndex(v) * 4 + j] = 1
                                    velocityMatrix[sizeMatrix - 1 - i][
                                        index - traci.vehicle.getLaneIndex(v) * 4 + j] = traci.vehicle.getSpeed(v)
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[sizeMatrix - 1 - startPos][
                                    index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1
                                velocityMatrix[sizeMatrix - 1 - startPos][
                                    index - traci.vehicle.getLaneIndex(v) * 4 + i] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[sizeMatrix - 1 - i][
                                        index - traci.vehicle.getLaneIndex(v) * 4 + j] = 1
                                    velocityMatrix[sizeMatrix - 1 - i][
                                        index - traci.vehicle.getLaneIndex(v) * 4 + j] = traci.vehicle.getSpeed(v)

        offset = cellLength * sizeLaneMatric
        for v in vehicles_road3_out:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = offset - traci.vehicle.getLanePosition(v)
                if std >= 0:
                    temp = (std + traci.vehicle.getLength(v)) / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (
                            traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][
                            sizeMatrix - 1 - endPos] = 1
                        velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][
                            sizeMatrix - 1 - endPos] = traci.vehicle.getSpeed(v)
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][
                                sizeMatrix - 1 - endPos] = 1
                            velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][
                                sizeMatrix - 1 - endPos] = traci.vehicle.getSpeed(v)
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yEndPos][sizeMatrix - 1 - endPos] = 1
                        velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yEndPos][
                            sizeMatrix - 1 - endPos] = traci.vehicle.getSpeed(v)
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][sizeMatrix - 1 - endPos] = 1
                            velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][
                                sizeMatrix - 1 - endPos] = traci.vehicle.getSpeed(v)
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][sizeMatrix - 1 - endPos] = 1
                            velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][
                                sizeMatrix - 1 - endPos] = traci.vehicle.getSpeed(v)

                    temp = (std) / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0
                        if temp_y_start < 0:
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][sizeMatrix - 1 - startPos] = 1
                            velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][
                                sizeMatrix - 1 - startPos] = traci.vehicle.getSpeed(v)
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][sizeMatrix - 1 - startPos] = 1
                                velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][
                                    sizeMatrix - 1 - startPos] = traci.vehicle.getSpeed(v)
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][sizeMatrix - 1 - startPos] = 1
                            velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][
                                sizeMatrix - 1 - startPos] = traci.vehicle.getSpeed(v)
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][sizeMatrix - 1 - startPos] = 1
                                velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][
                                    sizeMatrix - 1 - startPos] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + j][sizeMatrix - 1 - i] = 1
                                    velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + j][
                                        sizeMatrix - 1 - i] = traci.vehicle.getSpeed(v)
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][sizeMatrix - 1 - startPos] = 1
                                velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][
                                    sizeMatrix - 1 - startPos] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + j][sizeMatrix - 1 - i] = 1
                                    velocityMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + j][
                                        sizeMatrix - 1 - i] = traci.vehicle.getSpeed(v)

        # offset = cellLength * sizeLaneMatric
        for v in vehicles_road4_out:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = offset - traci.vehicle.getLanePosition(v)
                if std >= 0:
                    # if traci.vehicle.getVehicleClass(v) == 'taxi':
                    temp = (std + traci.vehicle.getLength(v)) / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[endPos][index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                        velocityMatrix[endPos][
                            index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = traci.vehicle.getSpeed(v)
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[endPos][index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                            velocityMatrix[endPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = traci.vehicle.getSpeed(v)
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[endPos][index - traci.vehicle.getLaneIndex(v) * 4 + yEndPos] = 1
                        velocityMatrix[endPos][
                            index - traci.vehicle.getLaneIndex(v) * 4 + yEndPos] = traci.vehicle.getSpeed(v)
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[endPos][index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1
                            velocityMatrix[endPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + i] = traci.vehicle.getSpeed(v)
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[endPos][index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1
                            velocityMatrix[endPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + i] = traci.vehicle.getSpeed(v)

                    temp = std / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0

                        if temp_y_start < 0:
                            positionMatrix[startPos][index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                            velocityMatrix[startPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = traci.vehicle.getSpeed(v)
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[startPos][index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                                velocityMatrix[startPos][
                                    index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = traci.vehicle.getSpeed(v)
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[startPos][index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                            velocityMatrix[startPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = traci.vehicle.getSpeed(v)
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[startPos][index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1
                                velocityMatrix[startPos][
                                    index - traci.vehicle.getLaneIndex(v) * 4 + i] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[i][index - traci.vehicle.getLaneIndex(v) * 4 + j] = 1
                                    velocityMatrix[i][
                                        index - traci.vehicle.getLaneIndex(v) * 4 + j] = traci.vehicle.getSpeed(v)
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[startPos][index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1
                                velocityMatrix[startPos][
                                    index - traci.vehicle.getLaneIndex(v) * 4 + i] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[i][index - traci.vehicle.getLaneIndex(v) * 4 + j] = 1
                                    velocityMatrix[i][
                                        index - traci.vehicle.getLaneIndex(v) * 4 + j] = traci.vehicle.getSpeed(v)
        index = 14
        # offset = cellLength * sizeLaneMatric
        for v in vehicles_road1_out:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = offset - traci.vehicle.getLanePosition(v)
                if std >= 0:
                    temp = (std + traci.vehicle.getLength(v)) / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (
                            traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][endPos] = 1
                        velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][
                            endPos] = traci.vehicle.getSpeed(v)
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][endPos] = 1
                            velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][
                                endPos] = traci.vehicle.getSpeed(v)
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yEndPos][endPos] = 1
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][endPos] = 1
                            velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][
                                endPos] = traci.vehicle.getSpeed(v)
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][endPos] = 1
                            velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][
                                endPos] = traci.vehicle.getSpeed(v)

                    temp = (std) / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0
                        if temp_y_start < 0:
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][startPos] = 1
                            velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][
                                startPos] = traci.vehicle.getSpeed(v)
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][startPos] = 1
                                velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][
                                    startPos] = traci.vehicle.getSpeed(v)
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][startPos] = 1
                            velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][
                                startPos] = traci.vehicle.getSpeed(v)
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][startPos] = 1
                                velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][
                                    startPos] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j][i] = 1
                                    velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j][
                                        i] = traci.vehicle.getSpeed(v)
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][startPos] = 1
                                velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][
                                    startPos] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j][i] = 1
                                    velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j][
                                        i] = traci.vehicle.getSpeed(v)

        for v in vehicles_road2_out:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = offset - traci.vehicle.getLanePosition(v)
                if std >= 0:
                    # if traci.vehicle.getVehicleClass(v) == 'taxi':
                    temp = (std + traci.vehicle.getLength(v)) / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (
                            traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (
                            traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[sizeMatrix - 1 - endPos][
                            index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = 1
                        velocityMatrix[sizeMatrix - 1 - endPos][
                            index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = traci.vehicle.getSpeed(v)
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[sizeMatrix - 1 - endPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = 1
                            velocityMatrix[sizeMatrix - 1 - endPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = traci.vehicle.getSpeed(v)
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[sizeMatrix - 1 - endPos][
                            index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yEndPos] = 1
                        velocityMatrix[sizeMatrix - 1 - endPos][
                            index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yEndPos] = traci.vehicle.getSpeed(v)
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[sizeMatrix - 1 - endPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = 1
                            velocityMatrix[sizeMatrix - 1 - endPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = traci.vehicle.getSpeed(v)
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[sizeMatrix - 1 - endPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = 1
                            velocityMatrix[sizeMatrix - 1 - endPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = traci.vehicle.getSpeed(v)

                    temp = (std) / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0

                        if temp_y_start < 0:
                            positionMatrix[sizeMatrix - 1 - startPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = 1
                            velocityMatrix[sizeMatrix - 1 - startPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = traci.vehicle.getSpeed(v)
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[sizeMatrix - 1 - startPos][
                                    index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = 1
                                velocityMatrix[sizeMatrix - 1 - startPos][
                                    index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = traci.vehicle.getSpeed(
                                    v)
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[sizeMatrix - 1 - startPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = 1
                            velocityMatrix[sizeMatrix - 1 - startPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = traci.vehicle.getSpeed(v)
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[sizeMatrix - 1 - startPos][
                                    index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = 1
                                velocityMatrix[sizeMatrix - 1 - startPos][
                                    index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[sizeMatrix - 1 - i][
                                        index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j] = 1
                                    velocityMatrix[sizeMatrix - 1 - i][
                                        index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j] = traci.vehicle.getSpeed(v)
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[sizeMatrix - 1 - startPos][
                                    index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = 1
                                velocityMatrix[sizeMatrix - 1 - startPos][
                                    index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[sizeMatrix - 1 - i][
                                        index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j] = 1
                                    velocityMatrix[sizeMatrix - 1 - i][
                                        index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j] = traci.vehicle.getSpeed(v)

        offset = 210.62 - cellLength * sizeLaneMatric
        for v in vehicles_road3_in:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = traci.vehicle.getLanePosition(v) - offset
                if std >= 0:
                    temp = std / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (
                            traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][
                            sizeMatrix - 1 - endPos] = 1
                        velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][
                            sizeMatrix - 1 - endPos] = traci.vehicle.getSpeed(v)
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][
                                sizeMatrix - 1 - endPos] = 1
                            velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][
                                sizeMatrix - 1 - endPos] = traci.vehicle.getSpeed(v)
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yEndPos][
                            sizeMatrix - 1 - endPos] = 1
                        velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yEndPos][
                            sizeMatrix - 1 - endPos] = traci.vehicle.getSpeed(v)
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][
                                sizeMatrix - 1 - endPos] = 1
                            velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][
                                sizeMatrix - 1 - endPos] = traci.vehicle.getSpeed(v)
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][
                                sizeMatrix - 1 - endPos] = 1
                            velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][
                                sizeMatrix - 1 - endPos] = traci.vehicle.getSpeed(v)

                    temp = (std - traci.vehicle.getLength(v)) / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0
                        if temp_y_start < 0:
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][
                                sizeMatrix - 1 - startPos] = 1
                            velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][
                                sizeMatrix - 1 - startPos] = traci.vehicle.getSpeed(v)
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][
                                    sizeMatrix - 1 - startPos] = 1
                                velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][
                                    sizeMatrix - 1 - startPos] = traci.vehicle.getSpeed(v)
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][
                                sizeMatrix - 1 - startPos] = 1
                            velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos][
                                sizeMatrix - 1 - startPos] = traci.vehicle.getSpeed(v)
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][
                                    sizeMatrix - 1 - startPos] = 1
                                velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][
                                    sizeMatrix - 1 - startPos] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j][
                                        sizeMatrix - 1 - i] = 1
                                    velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j][
                                        sizeMatrix - 1 - i] = traci.vehicle.getSpeed(v)
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][
                                    sizeMatrix - 1 - startPos] = 1
                                velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i][
                                    sizeMatrix - 1 - startPos] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j][
                                        sizeMatrix - 1 - i] = 1
                                    velocityMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j][
                                        sizeMatrix - 1 - i] = traci.vehicle.getSpeed(v)

        offset = 210.62 - cellLength * sizeLaneMatric
        for v in vehicles_road4_in:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = traci.vehicle.getLanePosition(v) - offset
                if std >= 0:
                    # if traci.vehicle.getVehicleClass(v) == 'taxi':
                    temp = std / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (
                            traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (
                            traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[endPos][index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = 1
                        velocityMatrix[endPos][index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = traci.vehicle.getSpeed(v)
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[endPos][index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = 1
                            velocityMatrix[endPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = traci.vehicle.getSpeed(v)
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[endPos][index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yEndPos] = 1
                        velocityMatrix[endPos][
                            index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yEndPos] = traci.vehicle.getSpeed(v)
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[endPos][index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = 1
                            velocityMatrix[endPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = traci.vehicle.getSpeed(v)
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[endPos][index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = 1
                            velocityMatrix[endPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = traci.vehicle.getSpeed(v)

                    temp = (std - traci.vehicle.getLength(v)) / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0

                        if temp_y_start < 0:
                            positionMatrix[startPos][index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = 1
                            velocityMatrix[startPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = traci.vehicle.getSpeed(v)
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[startPos][index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = 1
                                velocityMatrix[startPos][
                                    index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = traci.vehicle.getSpeed(
                                    v)
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[startPos][index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = 1
                            velocityMatrix[startPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3 - yStartPos] = traci.vehicle.getSpeed(v)
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[startPos][index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = 1
                                velocityMatrix[startPos][
                                    index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[i][index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j] = 1
                                    velocityMatrix[i][
                                        index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j] = traci.vehicle.getSpeed(v)
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[startPos][index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = 1
                                velocityMatrix[startPos][
                                    index + traci.vehicle.getLaneIndex(v) * 4 + 3 - i] = traci.vehicle.getSpeed(v)
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[i][index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j] = 1
                                    velocityMatrix[i][
                                        index + traci.vehicle.getLaneIndex(v) * 4 + 3 - j] = traci.vehicle.getSpeed(v)

        # for v in positionMatrix:
        #     print(v)
        # for i in range(0,60):
        #     s = ''
        #     for j in range(0, 60):
        #         s+=str(positionMatrix[i][j])
        #     print s

        # for v in velocityMatrix:
        #     print(v)

        outputMatrix = [positionMatrix, velocityMatrix]
        output = np.transpose(outputMatrix)  # np.array(outputMatrix)
        output = output.reshape(1, 60, 60, 2)

        tentative_action_matrix = tentative_action[action]

        return [output, I], tentative_action_matrix

    def cal_yellow_phase(self, id_list, a_dec):
        v_on_road = []
        for id in id_list:
            vehicles_road = traci.edge.getLastStepVehicleIDs(id)
            for vehicle in vehicles_road:
                v_on_road.append(traci.vehicle.getSpeed(vehicle))
        if not v_on_road:
            v_on_road.append(0)

        return int(np.amax(v_on_road)/a_dec)
