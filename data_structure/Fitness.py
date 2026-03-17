from SimulatedAnnealing.Input import Input
from SimulatedAnnealing.Solution import Solution
from typing import List
import numpy as np

class Fitness:
    def __init__(self, solution: Solution, input: Input, weights: List[int]):
        self.objectiveMatrix = self.buildObjectiveMatrix(solution, input)
        self.weights = np.matrix(weights)
        self.objective = 0
        self.penalty = 0
        self.updatePenalty(solution, input)
        self.calculateObjective()

    def calculateF1(self, input: Input, solution: Solution, patientId: int):
        patientInfo = input.patientInfo[patientId]
        value = 0
        if solution.startDays[patientId] >= patientInfo['dMin']:
            value = patientInfo['cost'] * (solution.startDays[patientId] - patientInfo['dMin'])
        return value

    def updateF1(self, input: Input, solution: Solution, patientId: int):
        value = self.calculateF1(input, solution, patientId)
        self.updateObjectiveMatrix(patientId, 0, value)

    def calculateF2(self, input: Input, solution: Solution, patientId: int):
        patientInfo = input.patientInfo[patientId]
        value = 0
        if solution.startDays[patientId] >= patientInfo['dTarget']:
            value = patientInfo['cost'] * (solution.startDays[patientId] - patientInfo['dTarget'])
        return value
    
    def updateF2(self, input: Input, solution: Solution, patientId: int):
        value = self.calculateF2(input, solution, patientId)
        self.updateObjectiveMatrix(patientId, 1, value)

    def calculateF3(self, input: Input, solution: Solution, patientId: int):
        machineColumn = solution.patientAppointments[:,patientId,0]
        value = 0
        for ind in range(len(machineColumn)-1):
            if machineColumn[ind] is not None and machineColumn[ind+1] is not None:
                value += 1 if machineColumn[ind]%input.timeWindowsQty != machineColumn[ind+1]%input.timeWindowsQty else 0
        return value
    
    def updateF3(self, input: Input, solution: Solution, patientId: int):
        value = self.calculateF3(input, solution, patientId)
        self.updateObjectiveMatrix(patientId, 2, value)

    def calculateF4(self, input: Input, solution: Solution, patientId: int):
        machineColumn = solution.patientAppointments[:,patientId,0]
        value = 0
        twPref = input.patientInfo[patientId]['twPref']
        if twPref is not None:
            for machineTwId in machineColumn:
                if machineTwId is not None:
                    value += abs(machineTwId%input.timeWindowsQty - twPref)
        return value
    
    def updateF4(self, input: Input, solution: Solution, patientId: int):
        value = self.calculateF4(input, solution, patientId)
        self.updateObjectiveMatrix(patientId, 3, value)

    def calculateF5(self, input: Input, solution: Solution, patientId: int):
        machineColumn = solution.patientAppointments[:,patientId,0]
        value = sum(
            1 if input.machineEligibility[patientId][machineTwId//input.timeWindowsQty] != 2 else 0
            for machineTwId in machineColumn
            if machineTwId is not None
        )
        return value
    
    def updateF5(self, input: Input, solution: Solution, patientId: int):
        value = self.calculateF5(input, solution, patientId)
        self.updateObjectiveMatrix(patientId, 4, value)

    def calculateF6(self, input: Input, solution: Solution, patientId: int):
        machineColumn = solution.patientAppointments[:,patientId,0]
        value = 0
        for ind in range(len(machineColumn)-1):
            if machineColumn[ind] is not None and machineColumn[ind+1] is not None:
                value += 1 if input.machineBeamMatching[machineColumn[ind]//input.timeWindowsQty][machineColumn[ind+1]//input.timeWindowsQty] != 2 else 0
        return value

    def updateF6(self, input: Input, solution: Solution, patientId: int):
        value = self.calculateF6(input, solution, patientId)
        self.updateObjectiveMatrix(patientId, 5, value)

    def updateObjectiveMatrix(self, row, column, value):
        old_value = self.objectiveMatrix[row,column]
        self.objectiveMatrix[row,column] = value
        self.objective += (value-old_value)*int(self.weights[:,column])

    def calculateObjective(self):
        self.objective = 1 + (self.objectiveMatrix*self.weights.transpose()).sum() + self.penalty

    def updatePenalty(self, solution: Solution, input: Input):
        penaltyCapacity = 100
        penaltyProtocol = 100
        self.objective -= self.penalty
        
        self.penalty = -penaltyCapacity * solution.residualCapacity[solution.residualCapacity < 0].sum()
        #self.penalty = sum(cap * -penaltyCapacity for daysCapacity in solution.residualCapacity for cap in daysCapacity if cap < 0)
        #self.penalty += sum(int(solution.startDays[patientsGrouped[pgInd]] > solution.startDays[patientsGrouped[pgInd+1]]) * penaltyProtocol for patientsGrouped in input.patientsGroupedByProtocol.values() for pgInd in range(len(patientsGrouped)-1))
        self.objective += self.penalty


    def buildObjectiveMatrix(self, solution: Solution, input: Input):
        objMatrix = np.zeros((input.patientsQty, 6))
        for patientId in input.patientsIdList:
            objMatrix[patientId] = [
                self.calculateF1(input, solution, patientId),
                self.calculateF2(input, solution, patientId),
                self.calculateF3(input, solution, patientId),
                self.calculateF4(input, solution, patientId),
                self.calculateF5(input, solution, patientId),
                self.calculateF6(input, solution, patientId)
            ]
        return np.matrix(objMatrix)