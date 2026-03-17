import numpy as np
import copy
from SimulatedAnnealing.Input import Input
import pickle as pk

class Solution:
    def __init__(self, input, fileName, data, calculateCapacity= True):
        if fileName != None:
            with open(fileName, "rb") as input_file:
                data = pk.load(input_file)

        self.patientAppointments = np.array(data['patientAppointments'])
        self.startDays = data['startDays']
        #self.maxDays = max([input.patientInfo[ind_p]['numFractions']+start for ind_p, start in enumerate(self.startDays)]) + 10
        
        if calculateCapacity:
            self.residualCapacity = self.calculateResidualCapacity(input)
        else:
            self.residualCapacity = copy.deepcopy(input.machinesCapacity)

    def checkLegalStartDay(self, input, patientId, startDay):
        patientInfo = input.patientInfo[patientId]
        return startDay >= patientInfo['dMin'] and startDay <= (len(self.patientAppointments) - patientInfo['numFractions']) and patientInfo['allowedDays'][startDay] == 1

    def checkIfLegal(self, input: Input):
        # TODO the check can stop at the first illegal constraint encountered
        # TODO avoid checking things which sould be avoided by construction
        for patientId, patient in enumerate(self.patientAppointments.transpose()[0]):
            startDay = self.startDays[patientId]
            patientInfo = input.patientInfo[patientId]
            # consecutivness
            if sum(int(patient[treatDay] is None) for treatDay in range(startDay, startDay+patientInfo['numFractions'])) > 0:
                print(f"consecutiveness contraint violated for patient {patientId}")
                return False
            # minimum start day
            if not (startDay >= patientInfo['dMin']):
                print(f"minimumStartDay contraint violated for patient {patientId}, start day {startDay}, dmin {patientInfo['dMin']}")
                return False
            # maximumStartDay
            if not (startDay <= (len(self.patientAppointments) - patientInfo['numFractions'])):
                print(f"maximumStartDay contraint violated for patient {patientId}")
                return False
            # allowedStartDay
            if not (patientInfo['allowedDays'][startDay] == 1):
                print(f"allowedStartDay contraint violated for patient {patientId}")
                return False
            # allowedMachines
            if sum(int(input.machineEligibility[patientId][dayAssignment // input.timeWindowsQty] == 0) for dayAssignment in patient if dayAssignment is not None) > 0:
                print(f"allowedMachines contraint violated for patient {patientId}")
                return False
            
        # capacity 
        if sum(tw for day in self.residualCapacity for tw in day if tw < 0) > 0:
            print(f"capacity contraint violated")
            return False

        for patientsGrouped in input.patientsGroupedByProtocol.values():
            for pgInd in range(len(patientsGrouped)-1):
                p1 = patientsGrouped[pgInd]
                p2 = patientsGrouped[pgInd+1]
                if input.patientInfo[p1]['dMin'] == input.patientInfo[p2]['dMin'] and self.startDays[p1] > self.startDays[p2]:
                    print(f"sameProtocolPrecedence contraint violated")
                    print(f"DEBUG p1 = {p1}, p2 = {p2}, startDay1 = {self.startDays[p1]}, startDay2 = {self.startDays[p2]}")
                    return False
        # sameProtocolPrecedence = 0 == sum(int(self.startDays[patientsGrouped[pgInd]] > self.startDays[patientsGrouped[pgInd+1]]) for patientsGrouped in input.patientsGroupedByProtocol.values() for pgInd in range(len(patientsGrouped)-1))
        return True

    def calculateResidualCapacity(self, input: Input):
        residual = copy.deepcopy(input.machinesCapacity)
        for day in range(len(self.patientAppointments)): #range(self.maxDays):
            for patient in self.patientAppointments[day]:
                if patient[0] is not None:
                    residual[day][patient[0]] -= patient[1]
        return residual