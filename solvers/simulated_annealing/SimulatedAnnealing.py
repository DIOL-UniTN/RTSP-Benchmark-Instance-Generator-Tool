from data_structure.Input import Input
from data_structure.Solution import Solution
from data_structure.Fitness import Fitness
import numpy as np
import copy
from math import exp

MAX_ITER = 50
DEFAULT_NEIGHBOUR_PROB = [0.2 for _ in range(5)]
NEIGHBOUR_SIZE_KEYS = ['twDays', 'twShift', 'machineDays', 'machineShift']

class SimulatedAnnealing():
    def __init__(
        self,
        inputFile: str,
        solutionFile: str,
        weights,
        neighbourSizes,
        inputData=None,
        data=None,
        calculateCapacity=True,
        capacityPenaltyWeight=100,
    ):
        if inputData == None:
            self.input = Input(inputFile)
        else:
            self.input = inputData

        self.solution = Solution(self.input, solutionFile, data, calculateCapacity)

        # self.input.machinesCapacity = self.input.machinesCapacity[:self.solution.maxDays]
        # for info in self.input.patientInfo:
        #     info['allowedDays'] = info['allowedDays'][:self.solution.maxDays]
    
        self.fitness = Fitness(
            self.solution,
            self.input,
            weights,
            capacity_penalty_weight=capacityPenaltyWeight,
        )
        self.neighbourSizes = self._normalize_neighbour_sizes(neighbourSizes)
        self.capacityPenaltyWeight = capacityPenaltyWeight
        self._reset_move_tracking()

    def _normalize_neighbour_sizes(self, neighbourSizes):
        if isinstance(neighbourSizes, dict):
            return neighbourSizes

        if len(neighbourSizes) != len(NEIGHBOUR_SIZE_KEYS):
            raise ValueError(
                f"Expected {len(NEIGHBOUR_SIZE_KEYS)} neighbour sizes, got {len(neighbourSizes)}."
            )

        return {
            key: int(value)
            for key, value in zip(NEIGHBOUR_SIZE_KEYS, neighbourSizes)
        }

    def _reset_move_tracking(self):
        self.all_residual_cap_moves = []
        self.all_patient_appoint_moves = []
        self.all_start_day_moves = []
        self.all_old_obj_values = []
        self.acceptedMovesCounter = [0 for _ in range(5)]
        self.improvingMovesCounter = [0 for _ in range(5)]
        self.acceptedMovesGain = [0 for _ in range(5)]
        self.movesCount = [0 for _ in range(5)]

    def _handle_candidate(self, perturbation_index, former_obj, temperature):
        self.movesCount[perturbation_index] += 1

        accepted = self.acceptNew(former_obj, self.fitness.objective, temperature)
        improved = accepted and self.fitness.objective < former_obj

        if accepted:
            self.acceptedMovesCounter[perturbation_index] += 1
            if improved:
                self.improvingMovesCounter[perturbation_index] += 1
                self.acceptedMovesGain[perturbation_index] += former_obj - self.fitness.objective

        return accepted, improved

    def _store_accepted_moves(self, residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values):
        self.all_residual_cap_moves += residual_cap_moves
        self.all_patient_appoint_moves += patient_appoint_moves
        self.all_start_day_moves += start_day_moves
        self.all_old_obj_values += old_obj_values

    def shiftTimeWindow(self):
        #TODO for now it's only forward
        residual_cap_moves = []
        patient_appoint_moves = []
        start_day_moves = []
        old_obj_values = []

        #patient = np.random.randint(self.input.patientsQty)
        patient = np.random.choice(self.input.patientsIdList)

        if min(self.input.patientInfo[patient]['numFractions'], self.neighbourSizes['twDays']) == 1:
            days_size = 1
        else:
            days_size = np.random.randint(1, min(self.input.patientInfo[patient]['numFractions'], self.neighbourSizes['twDays']))
        days = np.random.randint(
            self.solution.startDays[patient], 
            self.solution.startDays[patient]+self.input.patientInfo[patient]['numFractions'],
            days_size
        )
        twQty = self.input.timeWindowsQty
        for day in days:
            current_ind = self.solution.patientAppointments[day][patient][0]
            current_machine = current_ind // twQty
            if self.neighbourSizes['twShift'] == 1:
                twShift = 1
            else:
                twShift = np.random.randint(1, self.neighbourSizes['twShift'])
                
            new_ind = ((current_ind + twShift) % twQty) + (current_machine * twQty)

            #update residual capacity
            old_ind = self.solution.patientAppointments[day][patient][0]
            residual_cap_moves.append({
                'row': day,
                'column': old_ind,
                'value': self.solution.residualCapacity[day][old_ind]
            })
            residual_cap_moves.append({
                'row': day,
                'column': new_ind,
                'value': self.solution.residualCapacity[day][new_ind]
            })
            self.solution.residualCapacity[day][old_ind] += self.solution.patientAppointments[day][patient][1]
            self.solution.residualCapacity[day][new_ind] -= self.solution.patientAppointments[day][patient][1]

            #update patient assignment
            patient_appoint_moves.append({
                'row': day,
                'column': patient,
                'value': [old_ind, self.solution.patientAppointments[day][patient][1]]
            })
            self.solution.patientAppointments[day][patient][0] = new_ind

            #update fitness
            old_obj_values.append({
                'row': patient,
                'column': 2,
                'value': self.fitness.objectiveMatrix[patient,2]
            })
            old_obj_values.append({
                'row': patient,
                'column': 3,
                'value': self.fitness.objectiveMatrix[patient,3]
            })
            self.fitness.updateF3(self.input, self.solution, patient)
            self.fitness.updateF4(self.input, self.solution, patient)

        return residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values

    def shiftMachine(self):
        #TODO for now it's only forward
        residual_cap_moves = []
        patient_appoint_moves = []
        start_day_moves = []
        old_obj_values = []

        twQty = self.input.timeWindowsQty
        patient = None
        machineDaysInd = 0
        alreadyChosenDays = []
        if self.neighbourSizes['machineDays'] == 1:
            days_size = 1
        else:
            days_size = np.random.randint(1, self.neighbourSizes['machineDays'])

        if self.neighbourSizes['machineShift'] == 1:
            machineShift = 1
        else:
            machineShift = np.random.randint(1, self.neighbourSizes['machineShift'])

        iterations = 0
        while machineDaysInd < days_size:
            if(patient != None and machineDaysInd >= self.input.patientInfo[patient]['numFractions']):
                break
            current_machine = 1
            new_machine = 1
            day = None
            while new_machine == current_machine:
                iterations += 1
                if machineDaysInd == 0:
                    #patient = np.random.randint(self.input.patientsQty)
                    patient = np.random.choice(self.input.patientsIdList)
                    
                day = np.random.randint(
                    self.solution.startDays[patient], 
                    self.solution.startDays[patient]+self.input.patientInfo[patient]['numFractions']
                )
                if day in alreadyChosenDays:
                    break
                current_ind = self.solution.patientAppointments[day][patient][0]
                current_machine = current_ind // twQty
                new_machine = (current_machine + machineShift) % self.input.machinesQty
                while self.input.machineEligibility[patient][new_machine] == 0:
                    new_machine = (new_machine + 1) % self.input.machinesQty
                if iterations > MAX_ITER:
                    return residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values

            if day in alreadyChosenDays:
                continue

            alreadyChosenDays.append(day)
            machineDaysInd += 1
            current_window = current_ind % twQty
            new_ind = (new_machine * twQty) + current_window

            #update residual capacity
            old_ind = self.solution.patientAppointments[day][patient][0]
            residual_cap_moves.append({
                'row': day,
                'column': old_ind,
                'value': self.solution.residualCapacity[day][old_ind]
            })
            residual_cap_moves.append({
                'row': day,
                'column': new_ind,
                'value': self.solution.residualCapacity[day][new_ind]
            })
            self.solution.residualCapacity[day][old_ind] += self.solution.patientAppointments[day][patient][1]
            self.solution.residualCapacity[day][new_ind] -= self.solution.patientAppointments[day][patient][1]

            #update patient assignment
            patient_appoint_moves.append({
                'row': day,
                'column': patient,
                'value': [old_ind, self.solution.patientAppointments[day][patient][1]]
            })
            self.solution.patientAppointments[day][patient][0] = new_ind

            #update fitness
            old_obj_values.append({
                'row': patient,
                'column': 4,
                'value': self.fitness.objectiveMatrix[patient,4]
            })
            old_obj_values.append({
                'row': patient,
                'column': 5,
                'value': self.fitness.objectiveMatrix[patient,5]
            })
            self.fitness.updateF5(self.input, self.solution, patient)
            self.fitness.updateF6(self.input, self.solution, patient)

        return residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values

    def swapTimeWindows(self):
        residual_cap_moves = []
        patient_appoint_moves = []
        start_day_moves = []
        old_obj_values = []

        patientsList = []
        patients = []
        w1 = None
        w2 = None
        day = None
        iterations = 0
        while w1 == w2:
            iterations += 1
            while len(patientsList) <= 1:
                day = np.random.randint(len(self.solution.patientAppointments))
                patientsList = [patientId for patientId in range(len(self.solution.patientAppointments[day])) 
                                if self.solution.patientAppointments[day][patientId][0] is not None and patientId in self.input.patientsIdList]
            #patients = np.random.choice(patientsList, 2, False)
            random_indexes = np.random.randint(0, len(patientsList), size=2)
            patients = [patientsList.pop(random_indexes[0]), patientsList.pop(random_indexes[1] - 1)]
            w1 = self.solution.patientAppointments[day][patients[0]][0] % self.input.timeWindowsQty
            w2 = self.solution.patientAppointments[day][patients[1]][0] % self.input.timeWindowsQty

            if iterations > MAX_ITER:
                return residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values

        newInd1 = self.solution.patientAppointments[day][patients[0]][0] - w1 + w2
        newInd2 = self.solution.patientAppointments[day][patients[1]][0] - w2 + w1

        #update residual capacity
        old_ind_p1 = self.solution.patientAppointments[day][patients[0]][0]
        old_ind_p2 = self.solution.patientAppointments[day][patients[1]][0]
        residual_cap_moves.append({
            'row': day,
            'column': old_ind_p1,
            'value': self.solution.residualCapacity[day][old_ind_p1]
        })
        residual_cap_moves.append({
            'row': day,
            'column': newInd1,
            'value': self.solution.residualCapacity[day][newInd1]
        })
        residual_cap_moves.append({
            'row': day,
            'column': old_ind_p2,
            'value': self.solution.residualCapacity[day][old_ind_p2]
        })
        residual_cap_moves.append({
            'row': day,
            'column': newInd2,
            'value': self.solution.residualCapacity[day][newInd2]
        })
        self.solution.residualCapacity[day][old_ind_p1] += self.solution.patientAppointments[day][patients[0]][1]
        self.solution.residualCapacity[day][newInd1] -= self.solution.patientAppointments[day][patients[0]][1]
        self.solution.residualCapacity[day][old_ind_p2] += self.solution.patientAppointments[day][patients[1]][1]
        self.solution.residualCapacity[day][newInd2] -= self.solution.patientAppointments[day][patients[1]][1]

        #update patient assignment
        patient_appoint_moves.append({
            'row': day,
            'column': patients[0],
            'value': [old_ind_p1, self.solution.patientAppointments[day][patients[0]][1]]
        })
        patient_appoint_moves.append({
            'row': day,
            'column': patients[1],
            'value': [old_ind_p2, self.solution.patientAppointments[day][patients[1]][1]]
        })
        self.solution.patientAppointments[day][patients[0]][0] = newInd1
        self.solution.patientAppointments[day][patients[1]][0] = newInd2

        #update fitness
        for p in patients:
            old_obj_values.append({
                'row': p,
                'column': 2,
                'value': self.fitness.objectiveMatrix[p,2]
            })
            old_obj_values.append({
                'row': p,
                'column': 3,
                'value': self.fitness.objectiveMatrix[p,3]
            })
            self.fitness.updateF3(self.input, self.solution, p)
            self.fitness.updateF4(self.input, self.solution, p)

        return residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values

    def swapMachines(self):
        residual_cap_moves = []
        patient_appoint_moves = []
        start_day_moves = []
        old_obj_values = []

        patientsList = []
        patients = []
        m1 = None
        m2 = None
        day = None
        # warning, potential infinite loop
        iterations = 0
        while True:
            while m1 == m2:
                iterations += 1
                while len(patientsList) <= 1:
                    day = np.random.randint(len(self.solution.patientAppointments))
                    patientsList = [patientId for patientId in range(len(self.solution.patientAppointments[day])) 
                                    if self.solution.patientAppointments[day][patientId][0] is not None and patientId in self.input.patientsIdList]
                #patients = np.random.choice(patients, 2)
                random_indexes = np.random.randint(0, len(patientsList), size=2)
                patients = [patientsList.pop(random_indexes[0]), patientsList.pop(random_indexes[1] - 1)]
                m1 = self.solution.patientAppointments[day][patients[0]][0] // self.input.timeWindowsQty
                m2 = self.solution.patientAppointments[day][patients[1]][0] // self.input.timeWindowsQty

                if iterations > MAX_ITER:
                    return residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values
                
            if self.input.machineEligibility[patients[0]][m2] != 0 and self.input.machineEligibility[patients[1]][m1] != 0:
                break
            else:
                m1 = None
                m2 = None

        newInd1 = (m2 * self.input.timeWindowsQty) + self.solution.patientAppointments[day][patients[0]][0] % self.input.timeWindowsQty
        newInd2 = (m1 * self.input.timeWindowsQty) + self.solution.patientAppointments[day][patients[1]][0] % self.input.timeWindowsQty

        #update residual capacity
        old_ind_p1 = self.solution.patientAppointments[day][patients[0]][0]
        old_ind_p2 = self.solution.patientAppointments[day][patients[1]][0]
        residual_cap_moves.append({
            'row': day,
            'column': old_ind_p1,
            'value': self.solution.residualCapacity[day][old_ind_p1]
        })
        residual_cap_moves.append({
            'row': day,
            'column': newInd1,
            'value': self.solution.residualCapacity[day][newInd1]
        })
        residual_cap_moves.append({
            'row': day,
            'column': old_ind_p2,
            'value': self.solution.residualCapacity[day][old_ind_p2]
        })
        residual_cap_moves.append({
            'row': day,
            'column': newInd2,
            'value': self.solution.residualCapacity[day][newInd2]
        })
        self.solution.residualCapacity[day][old_ind_p1] += self.solution.patientAppointments[day][patients[0]][1]
        self.solution.residualCapacity[day][newInd1] -= self.solution.patientAppointments[day][patients[0]][1]
        self.solution.residualCapacity[day][old_ind_p2] += self.solution.patientAppointments[day][patients[1]][1]
        self.solution.residualCapacity[day][newInd2] -= self.solution.patientAppointments[day][patients[1]][1]

        #update patient assignment
        patient_appoint_moves.append({
            'row': day,
            'column': patients[0],
            'value': [old_ind_p1, self.solution.patientAppointments[day][patients[0]][1]]
        })
        patient_appoint_moves.append({
            'row': day,
            'column': patients[1],
            'value': [old_ind_p2, self.solution.patientAppointments[day][patients[1]][1]]
        })
        self.solution.patientAppointments[day][patients[0]][0] = newInd1
        self.solution.patientAppointments[day][patients[1]][0] = newInd2

        #update fitness
        for p in patients:
            old_obj_values.append({
                'row': p,
                'column': 4,
                'value': self.fitness.objectiveMatrix[p,4]
            })
            old_obj_values.append({
                'row': p,
                'column': 5,
                'value': self.fitness.objectiveMatrix[p,5]
            })
            self.fitness.updateF5(self.input, self.solution, p)
            self.fitness.updateF6(self.input, self.solution, p)

        return residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values

    def shiftStartDay(self):
        residual_cap_moves = []
        patient_appoint_moves = []
        start_day_moves = []
        old_obj_values = []

        patient = None
        shift = None
        iterations = 0
        while patient == None:
            iterations += 1
            shift = 0
            # patient = np.random.randint(self.input.patientsQty)
            patient = np.random.choice(self.input.patientsIdList)
            minDayProtocol = 0
            maxDayProtocol = len(self.solution.patientAppointments) - self.input.patientInfo[patient]['numFractions']
            for patientsGrouped in self.input.patientsGroupedByProtocol.values():
                if patient in patientsGrouped:
                    indPat = patientsGrouped.index(patient)
                    tempMin = [self.solution.startDays[p] for p in patientsGrouped[:indPat] if self.solution.startDays[p] != None]
                    minDayProtocol = max(tempMin) if tempMin != [] else minDayProtocol
                    tempMax = [self.solution.startDays[p] for p in patientsGrouped[indPat+1:] if self.solution.startDays[p] != None]
                    maxDayProtocol = min(tempMax) if tempMax != [] else maxDayProtocol
                    break
                
            while shift == 0 or self.input.patientInfo[patient]['allowedDays'][self.solution.startDays[patient]+shift] == 0 or self.solution.startDays[patient]+shift < minDayProtocol or self.solution.startDays[patient]+shift > maxDayProtocol:
                if self.solution.startDays[patient] == self.input.patientInfo[patient]['dMin']:
                    shift += 1
                elif self.solution.startDays[patient] == len(self.solution.patientAppointments) - self.input.patientInfo[patient]['numFractions']:
                    shift -= -1
                else:
                    shift = np.random.choice([shift-1, shift+1])
                if self.solution.startDays[patient]+shift > len(self.solution.patientAppointments) - self.input.patientInfo[patient]['numFractions'] or self.solution.startDays[patient]+shift > maxDayProtocol:
                    # print('shifted too much forward, picking new random patient')
                    patient = None
                    break
                if self.solution.startDays[patient]+shift < self.input.patientInfo[patient]['dMin'] or self.solution.startDays[patient]+shift < minDayProtocol:
                    # print('shifted too much backward, picking new random patient')
                    patient = None
                    break
            
            if iterations > MAX_ITER:
                return residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values

        # TODO check if swap between first and last is a possible choice instead of shifting all the assignments
        old_start_day = self.solution.startDays[patient]
        #TODO not optimized, I add some patient entries twice
        for day in range(
            min(old_start_day, old_start_day + shift), 
            max(old_start_day + self.input.patientInfo[patient]['numFractions'], old_start_day + self.input.patientInfo[patient]['numFractions'] + shift)
        ):
            patient_appoint_moves.append({
                'row': day,
                'column': patient,
                'value': [self.solution.patientAppointments[day][patient][0], self.solution.patientAppointments[day][patient][1]]
            })
            if day >= old_start_day and day < old_start_day + self.input.patientInfo[patient]['numFractions']:
                res_col = self.solution.patientAppointments[day][patient][0]
                residual_cap_moves.append({
                    'row': day,
                    'column': res_col,
                    'value': self.solution.residualCapacity[day][res_col]
                })
                residual_cap_moves.append({
                    'row': day+shift,
                    'column': res_col,
                    'value': self.solution.residualCapacity[day+shift][res_col]
                })
        #update residual capacity
        for day in range(old_start_day, old_start_day + self.input.patientInfo[patient]['numFractions']):
            appointment = self.solution.patientAppointments[day][patient]
            self.solution.residualCapacity[day][appointment[0]] += appointment[1]
            self.solution.residualCapacity[day+shift][appointment[0]] -= appointment[1]

        #update start day
        start_day_moves.append({
            'index': patient,
            'value': old_start_day
        })
        self.solution.startDays[patient] += shift

        #update patient assignment
        temp = self.solution.patientAppointments.transpose()
        temp[:, patient] = np.roll(temp[:,patient], shift)
        self.solution.patientAppointments = temp.transpose() #this should be done by default by python because temp is still refering to patientAppointments, it's not a copy

        #update fitness
        old_obj_values.append({
            'row': patient,
            'column': 0,
            'value': self.fitness.objectiveMatrix[patient, 0]
        })
        old_obj_values.append({
            'row': patient,
            'column': 1,
            'value': self.fitness.objectiveMatrix[patient, 1]
        })
        self.fitness.updateF1(self.input, self.solution, patient)
        self.fitness.updateF2(self.input, self.solution, patient)

        return residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values

    def swapStartDays(self):
        pass

    def neighbour(self, neighbourProb):
        perturbations = [
            self.shiftTimeWindow,
            self.shiftMachine,
            self.swapTimeWindows,
            self.swapMachines,
            self.shiftStartDay#,
            #self.swapStartDays
        ]
        pert_indexes = [i for i in range(len(perturbations))]
        func_ind = np.random.choice(pert_indexes, p=neighbourProb)
        func = perturbations[func_ind]
        residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values = func()
        self.fitness.updatePenalty(self.solution, self.input)
        return func_ind, residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values

    def acceptNew(self, e, e_new, T):
        if(e_new<=e):
            return True
        else:
            if (np.random.uniform(0, 1) < exp(-(e_new-e)/T)):
                #print("Delta worst acceptance:", e_new-e)
                return True
        return False

    def updateTemperature(self, T, alpha):
            T = alpha*T
            return T, alpha
    
    def undoMoves(self, residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values):
        for res_move in residual_cap_moves:
            self.solution.residualCapacity[res_move['row']][res_move['column']] = res_move['value']

        for pat_move in patient_appoint_moves:
            self.solution.patientAppointments[pat_move['row']][pat_move['column']] = pat_move['value']
            
        for start_move in start_day_moves:
            self.solution.startDays[start_move['index']] = start_move['value']
            
        for old_val in old_obj_values:
            self.fitness.updateObjectiveMatrix(old_val['row'], old_val['column'], old_val['value'])

        self.fitness.updatePenalty(self.solution, self.input)

    def doMoves(self, bestSolution: Solution, bestFitness: Fitness):
        for res_move in self.all_residual_cap_moves:
            bestSolution.residualCapacity[res_move['row']][res_move['column']] = self.solution.residualCapacity[res_move['row']][res_move['column']]

        for pat_move in self.all_patient_appoint_moves:
            bestSolution.patientAppointments[pat_move['row']][pat_move['column']] = self.solution.patientAppointments[pat_move['row']][pat_move['column']]

        for start_move in self.all_start_day_moves:
            bestSolution.startDays[start_move['index']] = self.solution.startDays[start_move['index']]

        for old_val in self.all_old_obj_values:
            bestFitness.updateObjectiveMatrix(old_val['row'], old_val['column'], self.fitness.objectiveMatrix[old_val['row'], old_val['column']])

        bestFitness.updatePenalty(bestSolution, self.input)
        self.all_residual_cap_moves = []
        self.all_patient_appoint_moves = []
        self.all_start_day_moves = []
        self.all_old_obj_values = []

        return bestSolution, bestFitness

    def main(self, startT, kMax, alpha, neighbourProb, seed, max_iter = 50):
        global MAX_ITER
        MAX_ITER = max_iter

        np.random.seed(seed)
        self._reset_move_tracking()
        T = startT
        historyAccepted = []
        historyBest = []
        # TODO look for "profiler" and if copy.deepcopy takes more time than undoing the perturbation
        bestSolution = copy.deepcopy(self.solution)
        bestFitness = copy.deepcopy(self.fitness)
        print('starting fitness:', self.fitness.objective)
        for _ in range(kMax):
            former_obj = self.fitness.objective # TODO check se objective viene passato per valore o per riferimento
            perturbation_index, residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values = self.neighbour(neighbourProb)

            accepted, _ = self._handle_candidate(perturbation_index, former_obj, T)

            if not accepted:
                self.undoMoves(residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values)
            else:
                self._store_accepted_moves(residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values)

            T, alpha = self.updateTemperature(T, alpha)

            historyAccepted.append(self.fitness.objective)

            if self.fitness.objective < bestFitness.objective:
                bestSolution, bestFitness = self.doMoves(bestSolution, bestFitness)

    
            historyBest.append(bestFitness.objective)

        # print("Temperature", T)
        return bestSolution, bestFitness, historyAccepted, historyBest
    
    def main_reheating(self, startT, kMax, alpha, reheating_iter, neighbourProb, seed, max_iter = 50):
        global MAX_ITER
        MAX_ITER = max_iter

        np.random.seed(seed)
        self._reset_move_tracking()
        historyAccepted = []
        historyBest = []
        # TODO look for "profiler" and if copy.deepcopy takes more time than undoing the perturbation
        bestSolution = copy.deepcopy(self.solution)
        bestFitness = copy.deepcopy(self.fitness)
        print(alpha)
        for iter in range(reheating_iter):
            T = startT
            for _ in range(int(kMax/reheating_iter)):
                former_obj = self.fitness.objective # TODO check se objective viene passato per valore o per riferimento
                perturbation_index, residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values = self.neighbour(neighbourProb)

                accepted, _ = self._handle_candidate(perturbation_index, former_obj, T)

                if not accepted:
                    self.undoMoves(residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values)
                else:
                    self._store_accepted_moves(residual_cap_moves, patient_appoint_moves, start_day_moves, old_obj_values)

                T, alpha = self.updateTemperature(T, alpha)

                historyAccepted.append(self.fitness.objective)

                if self.fitness.objective < bestFitness.objective:
                    bestSolution, bestFitness = self.doMoves(bestSolution, bestFitness)

        
                historyBest.append(bestFitness.objective)

        print("Temperature", T)
        import matplotlib.pyplot as plt

        plt.plot(historyAccepted)
        plt.ylabel('objective')
        plt.xlabel('iterations')
        plt.show()

        plt.plot(historyBest)
        plt.ylabel('objective')
        plt.xlabel('iterations')
        plt.show()
        return bestSolution, bestFitness, historyAccepted, historyBest

    def run(
        self,
        start_temperature=100.0,
        k_max=1000,
        cooling_rate=0.99,
        neighbour_prob=None,
        seed=42,
        max_iter=50,
    ):
        if neighbour_prob is None:
            neighbour_prob = DEFAULT_NEIGHBOUR_PROB

        bestSolution, bestFitness, historyAccepted, historyBest = self.main(
            start_temperature,
            k_max,
            cooling_rate,
            neighbour_prob,
            seed,
            max_iter=max_iter,
        )
        self.solution = bestSolution
        self.fitness = bestFitness
        return bestSolution, bestFitness, historyAccepted, historyBest
