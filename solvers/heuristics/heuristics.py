import copy
import time
from data_structure.Input import Input
from data_structure.Fitness import Fitness
from data_structure.Solution import Solution

def algorithmFirstFit(inputData, patientsBatch, windowSet, beamMatchedMachines, assignment, startDays):
    for patientsDaily in patientsBatch:
        for patient in patientsDaily:
            twPref = inputData.patientInfo[patient]['twPref']
            if twPref is not None:
                windowSet.remove(twPref)
                windowSet.insert(0, twPref)
            else:
                windowSet.sort()
            #print("patient: ", patient.id)
            # Iterazione sui pazienti
            #for k, patient in data["patients"].items():
            assigned_all_fractions = False  # Flag per verificare se tutte le frazioni del paziente sono state assegnate

            for start_day in range(inputData.patientInfo[patient]['dMin'], inputData.daysQty):
                if inputData.patientInfo[patient]['allowedDays'][start_day] == 0:
                    continue
                
                #TODO ordinare i gruppi di BM machines per capacità residua (e anche le macchine all'interno dei gruppi)
                #nuovo wrapper in cui ciclo sulle matched machine: se nono trovo tutto un assegnamento con un blocco di matched machines, provo con un altro blocco, prima di provare con un nuovo giorno
                
                temp_assignment_list = [None for _ in range(len(beamMatchedMachines))]
                temp_startDays_list = [None for _ in range(len(beamMatchedMachines))]
                temp_bin_capacity_list = [None for _ in range(len(beamMatchedMachines))]

                for ind_bm, matchedMachines in enumerate(beamMatchedMachines):
                    # machines = patient.getOrderedAllowedMachines(matchedMachines)
                    machines = [machine for machine in matchedMachines if inputData.machineEligibility[patient][machine] == 2]
                    machines += [machine for machine in matchedMachines if inputData.machineEligibility[patient][machine] == 1]
                    if len(machines) == 0:
                        continue
                    temp_assignment = copy.deepcopy(assignment)
                    temp_startDays = copy.deepcopy(startDays)
                    temp_bin_capacity = copy.deepcopy(inputData.machinesCapacity)
                    previous_day = None
                    success = True

                    # Iterazione sulle frazioni del paziente
                    for fraction in range(inputData.patientInfo[patient]['numFractions']):
                        assigned = False

                        # Cerca il primo giorno e macchina disponibili che rispettano i vincoli
                        for d in range(start_day, inputData.daysQty):
                            if fraction == 0 and inputData.patientInfo[patient]['allowedDays'][d] == 0:
                                continue

                            if (previous_day is not None and d != previous_day + 1):
                                continue

                            # probabilmente fare un if d > previous_day + 1 then break, per evitarsi cicli inutili se è andato oltre il giorno successivo
                            if (previous_day is not None and d > previous_day + 1) or d >= inputData.daysQty:
                                break
                            
                            for machine in machines:
                                for window in windowSet:
                                    # if machine.id in temp_bin_capacity[d] and temp_bin_capacity[d][machine.id][window] >= patient.getFractionLength(fraction): #temp_bin_capacity[d][machine] >= fraction
                                    # inputData.machinesCapacity[dayId][(machineId*windows_num)+windowId]
                                    fractionLenght = inputData.patientInfo[patient]['fractionsDuration'][fraction]
                                    if temp_bin_capacity[d][(machine*inputData.timeWindowsQty)+window] >= fractionLenght: 
                                        temp_assignment[d][patient] = [(machine*inputData.timeWindowsQty)+window, fractionLenght]
                                        if fraction == 0:
                                            temp_startDays[patient] = d
                                        #temp_assignment[(fraction, patient.id)] = (window, machine.id, d, fractionLenght)
                                        temp_bin_capacity[d][(machine*inputData.timeWindowsQty)+window] -= fractionLenght
                                        previous_day = d
                                        assigned = True
                                        break
                                if assigned == True:
                                    break
                            if assigned:
                                break

                        if not assigned:
                            success = False
                            break
                    if success:
                        temp_assignment_list[ind_bm] = temp_assignment
                        temp_startDays_list[ind_bm] = temp_startDays
                        temp_bin_capacity_list[ind_bm] = temp_bin_capacity
                        # break

                if success:  # Se tutte le frazioni sono state assegnate, aggiorna i dati effettivi
                    start_day_temp = inputData.daysQty + 1
                    winner = None
                    for ind_bm in range(len(beamMatchedMachines)):
                        if temp_startDays_list[ind_bm] != None and temp_startDays_list[ind_bm][patient] < start_day_temp:
                            start_day_temp = temp_startDays_list[ind_bm][patient]
                            winner = ind_bm

                    assignment = temp_assignment_list[winner]
                    startDays = temp_startDays_list[winner]
                    inputData.machinesCapacity = temp_bin_capacity_list[winner]
                    assigned_all_fractions = True
                    break
            
            if not assigned_all_fractions:
                print(f"Errore: Non è possibile assegnare tutte le frazioni del paziente {patient}.")
                print(f"  Giorno di arrivo: {inputData.patientInfo[patient]['dMin']}")
                return None, None
        
    return assignment, startDays

def heuristicFirstFit(inputData: Input):
    patientsBatch = [[patient for ind, patientsProtocol in inputData.patientsGroupedByProtocol.items() for patient in patientsProtocol if inputData.patientInfo[patient]['dMin'] == day] for day in range(inputData.daysQty)]
    patientsBatch = list(filter(None, patientsBatch))

    beamMatchedMachines = [[machineId for machineId in range(inputData.machinesQty) if inputData.machineBeamMatching[machineId][m2Id] != 0] for m2Id in range(inputData.machinesQty)]
    temp_beamMatchedMachines = []
    for el in beamMatchedMachines:
        if el not in temp_beamMatchedMachines:
            temp_beamMatchedMachines.append(el)
    beamMatchedMachines = temp_beamMatchedMachines   
    beamMatchedMachines.sort(key=lambda x: len(x), reverse=True)

    windowSet = [el for el in range(inputData.timeWindowsQty)]

    assignment = [[[None, None] for k in range(len(inputData.patientInfo))] for i in range(inputData.daysQty)]
    startDays = [None for k in range(len(inputData.patientInfo))]

    start_time = time.time()
    assignment, startDays = algorithmFirstFit(inputData, patientsBatch, windowSet, beamMatchedMachines, assignment, startDays)

    elapsed_time = time.time() - start_time

    solution = Solution(inputData, None, {'patientAppointments': assignment, 'startDays': startDays}, False)
    
    return solution, 'first_fit', elapsed_time

def algorithmBestFit(inputData, patientsDaily, windowSet, beamMatchedMachines, assignment, startDays):
    def assign_patient_fractions(patient, fractions, assignment, startDays, bin_capacity, start_day, machines, windowSet):
        previous_day = None
        for fraction in fractions:
            best_day = None
            best_machine = None
            best_window = None
            min_residual_capacity = float('inf')
            fractionLenght = inputData.patientInfo[patient]['fractionsDuration'][fraction]

            for d in range(start_day, inputData.daysQty):
                if d < inputData.patientInfo[patient]['dMin'] or (fraction == 0 and inputData.patientInfo[patient]['allowedDays'][d] == 0):
                    continue

                if previous_day is not None and d != previous_day + 1:
                    continue

                for machine in machines:
                    for window in windowSet:
                        if bin_capacity[d][(machine*inputData.timeWindowsQty)+window] >= fractionLenght:
                            residual_capacity = bin_capacity[d][(machine*inputData.timeWindowsQty)+window] - fractionLenght

                            if residual_capacity < min_residual_capacity:
                                best_day = d
                                best_machine = machine
                                best_window = window
                                min_residual_capacity = residual_capacity

                # Da mettere se devo confrontare solamente le macchine del primo giorno disponibile
                if best_day == d:
                    break

            if best_day is None or best_machine is None or best_window is None:
                return False

            previous_day = best_day
            assignment[best_day][patient] = [(best_machine*inputData.timeWindowsQty)+best_window, fractionLenght]
            if fraction == 0:
                startDays[patient] = best_day
            bin_capacity[best_day][(best_machine*inputData.timeWindowsQty)+best_window] -= fractionLenght
            start_day = best_day + 1  # Passa al giorno successivo

        return True

    def assign_with_backtrack(patient, fractions, assignment, startDays, windowSet):
        twPref = inputData.patientInfo[patient]['twPref']
        if twPref is not None:
            windowSet.remove(twPref)
            windowSet.insert(0, twPref)
        else:
            windowSet.sort()

        for start_day in range(inputData.patientInfo[patient]['dMin'], inputData.daysQty):
            if inputData.patientInfo[patient]['allowedDays'][start_day] == 0:
                continue

            for matchedMachines in beamMatchedMachines:
                #machines = patient.getOrderedAllowedMachines(matchedMachines)
                machines = [machine for machine in matchedMachines if inputData.machineEligibility[patient][machine] == 2]
                machines += [machine for machine in matchedMachines if inputData.machineEligibility[patient][machine] == 1]
                if len(machines) == 0:
                    continue

                temp_assignment = copy.deepcopy(assignment) #assignment.copy()
                temp_bin_capacity = copy.deepcopy(inputData.machinesCapacity) #{d: bin_capacity[d].copy() for d in bin_capacity}
                temp_startDays = copy.deepcopy(startDays)

                if assign_patient_fractions(patient, fractions, temp_assignment, temp_startDays, temp_bin_capacity, start_day, machines, windowSet):
                    assignment = copy.deepcopy(temp_assignment)
                    startDays = copy.deepcopy(temp_startDays)
                    inputData.machinesCapacity = copy.deepcopy(temp_bin_capacity)
                    return True, assignment, startDays

        return False, assignment, startDays
    
    
    for patient in patientsDaily:
        fractions = list(range(inputData.patientInfo[patient]['numFractions']))
        bool_value, assignment, startDays = assign_with_backtrack(patient, fractions, assignment, startDays, windowSet)
        if not bool_value:
            print(f"Errore: Non è possibile assegnare tutte le frazioni del paziente {patient}.")
            return None, None
    return assignment, startDays

def heuristicBestFit(inputData: Input):
    patientsBatch = [[patient for ind, patientsProtocol in inputData.patientsGroupedByProtocol.items() for patient in patientsProtocol if inputData.patientInfo[patient]['dMin'] == day] for day in range(inputData.daysQty)]
    patientsBatch = list(filter(None, patientsBatch))
    
    beamMatchedMachines = [[machineId for machineId in range(inputData.machinesQty) if inputData.machineBeamMatching[machineId][m2Id] != 0] for m2Id in range(inputData.machinesQty)]
    temp_beamMatchedMachines = []
    for el in beamMatchedMachines:
        if el not in temp_beamMatchedMachines:
            temp_beamMatchedMachines.append(el)
    beamMatchedMachines = temp_beamMatchedMachines   
    beamMatchedMachines.sort(key=lambda x: len(x), reverse=True)

    windowSet = [el for el in range(inputData.timeWindowsQty)]

    assignment = [[[None, None] for k in range(len(inputData.patientInfo))] for i in range(inputData.daysQty)]
    startDays = [None for k in range(len(inputData.patientInfo))]

    start_time = time.time()
    for patientsDaily in patientsBatch:
        assignment, startDays = algorithmBestFit(inputData, patientsDaily, windowSet, beamMatchedMachines, assignment, startDays)

    elapsed_time = time.time() - start_time

    solution = Solution(inputData, None, {'patientAppointments': assignment, 'startDays': startDays}, False)

    return solution, 'best_fit', elapsed_time


def heuristic(arguments, weightsAlpha):
    print('file: ', arguments.instance_file)
    objWeights = [w for w in weightsAlpha[arguments.obj].values()]

    inputFile = arguments.instance_file
    inputData = Input(inputFile)

    if arguments.func == 'heuristic_FF':
        solution, _, timeElapsed = heuristicFirstFit(inputData)
    elif arguments.func == 'heuristic_BF':
        solution, _, timeElapsed = heuristicBestFit(inputData)

    fitness = Fitness(solution, inputData, objWeights)
    f_obj = [row.sum() for row in fitness.objectiveMatrix.transpose()]

    sol_to_store = getSolutionToStore(solution, arguments)
    print("legal", solution.checkIfLegal(inputData))

    csvDict = {
        'obj': arguments.obj,
        'code': arguments.gen,
        'instance': arguments.instance,
        'time to optimum': timeElapsed,
        'objective': fitness.objective,
        'f1': f_obj[0],
        'f2': f_obj[1],
        'f3': f_obj[2],
        'f4': f_obj[3],
        'f5': f_obj[4],
        'f6': f_obj[5],
        'seed': arguments.seed
    }

    return {'pickle': sol_to_store, 'csv': csvDict}


def getSolutionToStore(bestSolution, args):
    solutionToStore = {
        'patientAppointments': bestSolution.patientAppointments,
        'startDays': bestSolution.startDays,
    }
    import pickle
    with open(f'SimulatedAnnealing/solutions/{args.heuristic}/{args.period}/{args.dataset_folder}/{args.gen}/{args.instance}.pk', 'wb') as handle:
        pickle.dump(solutionToStore, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return solutionToStore
