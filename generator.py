import pickle
import os
import argparse
import random
import numpy as np
import resource
import csv
import json
from SimulatedAnnealing.Input import Input
from Heuristic_new_dataStructure import heuristicFirstFit, getSolutionToStore


parser = argparse.ArgumentParser(description='Solve different instances of RTSP')


parser.add_argument('-t', dest='dataset_title', action='store',
                    help='title (name) of generated dataset', type=str, default='new_dataset_2')
parser.add_argument('-i', dest='instances_number', action='store',
                    help='number of instances to generate', type=int, default=50)
parser.add_argument('-s', dest='seed', action='store',
                    help='seed', type=int, default=198743)

parser.add_argument('-m', dest='machines_number', action='store',
                    help='number of machines', type=int, default=6)
parser.add_argument('-tw', dest='time_windows_number', action='store',
                    help='number of time windows', type=int, default=2)

parser.add_argument('-c', dest='capacities', action='store',
                    help='list of time windows capacities for each machine', nargs = '+', type=int, default=[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240])
parser.add_argument('-o', dest='initial_occupation_percentage', action='store',
                    help='list of time windows initial occupation for each machine suggested between 0.35 to 0.9', nargs = '+', type=float, default=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
parser.add_argument('-v', dest='occupation_decay_velocity', action='store',
                    help='list of time windows occupation decay velocity for each machine', nargs = '+', type=float, default=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

parser.add_argument('-bm', dest='min_beam_matched_sets', action='store',
                    help='minimum amount of sets of beam matched machines', type=int, default=0)
parser.add_argument('-pbm', dest='min_partial_beam_matched_sets', action='store',
                    help='minimum amount of sets of partial beam matched machines', type=int, default=0)

parser.add_argument('-l', dest='arrival_mean', action='store',
                    help='patient arrival mean', type=int, default=7)
parser.add_argument('-pp', dest='priorities_percentage', action='store',
                    help='percentage of patients priority', nargs = '+', type=float, default=[0.3, 0.3, 0.4])

args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)

with open('generator_parameter.json') as json_file:
    parameters = json.load(json_file)

time_horizon = 150
daysPatientsArrival = 25

weights = [10, 3, 1]
dayTarget = [2, 10, 20]

def get_random_tw_pref(num_tw):
        if num_tw < 2:
            return num_tw
        if random.uniform(0,1) <= 0.8:
            return np.random.choice(num_tw, p = [1/num_tw for ind in range(num_tw)])
        else:
            return None


protocolMachineEligibility = []
for protocol in parameters['patients']['protocols_info']:
    m_elig = np.zeros(args.machines_number)
    preferredLen = np.random.randint(1, round(args.machines_number*0.9)) if round(args.machines_number*0.9) > 1 else 1
    m_temp = np.random.choice(args.machines_number, preferredLen, replace = False)
    atLeastOnePreferred = False
    for m in m_temp:
        if atLeastOnePreferred and 0.6 < np.random.rand():
            m_elig[int(m)] = 1
        else:
            m_elig[int(m)] = 2
            atLeastOnePreferred = True 
    protocolMachineEligibility.append(m_elig)

# TODO check min_partial_beam_matched_sets < numMachines/2
# TODO check min_partial_beam_matched_sets >= min_beam_matched_sets
#completeSets = np.random.randint(0, round(args.machines_number * 0.2)) if round(args.machines_number * 0.2) > 0 else 0
partialSets = np.random.randint(args.min_partial_beam_matched_sets, round(args.machines_number * 0.3)) if round(args.machines_number * 0.3) > args.min_partial_beam_matched_sets else args.min_partial_beam_matched_sets
machineBeamMatching = np.zeros((args.machines_number, args.machines_number))
for ind in range(args.machines_number):
    machineBeamMatching[ind][ind] = 2

m_remaining = np.arange(args.machines_number)
completeSets = 0
for _ in range(partialSets):
    complete = False
    setLen = np.random.randint(2, args.machines_number//partialSets) if args.machines_number//partialSets > 2 else 2
    m_temp = np.random.choice(m_remaining, setLen, replace = False)
    for ind_m, m in enumerate(m_temp):
        m_remaining = np.delete(m_remaining, np.where(m_remaining == m))
        for m2 in m_temp[ind_m+1:]:
            if m != m2:
                if completeSets < args.min_beam_matched_sets or 0.8 < np.random.rand():
                    complete = True
                    machineBeamMatching[m][m2] = 2
                    machineBeamMatching[m2][m] = 2
                else:
                    machineBeamMatching[m][m2] = 1
                    machineBeamMatching[m2][m] = 1
    if complete == True:
        completeSets += 1
    complete = False

print('completeSets', completeSets)

for instance in range(args.instances_number):

    data = {
        'machineEligibility': [],
        'machineBeamMatching': machineBeamMatching,
        'patientInfo': [],
        'machinesCapacity': [[] for _ in range(time_horizon)],
        'timeWindowsQty': args.time_windows_number,
        'patientsQty': None,
        'machinesQty': args.machines_number,
        'machinesMaxCapacity': args.capacities[ind],
        'patientsGroupedByProtocol': None,
        'daysQty': time_horizon
    }

    orderedProtocol = list(range(len(parameters['patients']['protocols_info'])))
    orderedProtocol.sort(key=lambda protocolId: parameters['patients']['protocols_info'][str(protocolId)]['priority'])
    data['patientsGroupedByProtocol'] = {i: [] for i in orderedProtocol}


    for m in range(args.machines_number):
        for tw in range(args.time_windows_number):
            ind = m * args.time_windows_number + tw
            beta_0 = np.log(args.initial_occupation_percentage[ind] * args.capacities[ind])
            beta_1 = (1 - args.occupation_decay_velocity[ind]) * parameters['machines']['beta_1_min'] + args.occupation_decay_velocity[ind] * parameters['machines']['beta_1_max']
            beta_2 = parameters['machines']['beta_2_mean']
            for d in range(time_horizon):
                mu_d = np.exp(beta_0 + beta_1 * d + beta_2 * (d**2))
                k_d = parameters['machines']['k_d_median'][d] if len(parameters['machines']['k_d_median']) > abs(d) else 1000
                if k_d != 1000:
                    occ_d = np.random.negative_binomial(parameters['machines']['k_d_median'][d], parameters['machines']['k_d_median'][d]/(parameters['machines']['k_d_median'][d]+mu_d))
                else:
                    occ_d = np.random.poisson(mu_d)
                occ_d = min(args.capacities[ind], occ_d)
                data['machinesCapacity'][d].append(args.capacities[ind] - occ_d)



    patientId = 0

    for d in range(daysPatientsArrival):
        patients_num = np.random.poisson(args.arrival_mean)
        for _ in range(patients_num):
            priority = np.random.choice(
                np.arange(1, len(args.priorities_percentage) + 1),
                p=args.priorities_percentage
            )

            protocol = str(np.random.choice(
                np.array([int(el) for el in parameters['patients']['protocols_probabilities'][priority-1].keys()]),
                p=[float(el)/sum([float(el) for el in parameters['patients']['protocols_probabilities'][priority-1].values()]) for el in parameters['patients']['protocols_probabilities'][priority-1].values()]
            ))

            data['machineEligibility'].append(protocolMachineEligibility[int(protocol)])

            data['patientsGroupedByProtocol'][int(protocol)].append(patientId)

            data['patientInfo'].append({
                'cost': weights[priority-1],
                'dMin': d,
                'dTarget': d + dayTarget[priority-1],
                'twPref': get_random_tw_pref(args.time_windows_number),
                'numFractions': parameters['patients']['protocols_info'][protocol]['numFractions'],
                'allowedDays': [int(day+1 in parameters['patients']['protocols_info'][protocol]['allowed_start_weekdays_protocol']) for day in range(time_horizon)],
                'priority': priority,
                'fractionsDuration': [parameters['patients']['protocols_info'][protocol]['duration_first']] + [parameters['patients']['protocols_info'][protocol]['duration_others'] for _ in range(parameters['patients']['protocols_info'][protocol]['numFractions'] - 1)]
            })

            patientId += 1

    data['patientsQty'] = len(data['patientInfo'])

    for protocolId in orderedProtocol:
        data['patientsGroupedByProtocol'][protocolId].sort(key=lambda patientId: data['patientInfo'][patientId]['dTarget'])

    inputData = Input('', data = data)

    for obj in range(1,2):
        objWeights = [w for w in parameters['weightsAlpha'][obj].values()]
        sa, _, timeElapsed = heuristicFirstFit(inputData, objWeights)
        f_obj = [row.sum() for row in sa.fitness.objectiveMatrix.transpose()]

        sol_to_store = {
            'patientAppointments': sa.solution.patientAppointments,
            'startDays': sa.solution.startDays,
            'acceptedMovesCounter': sa.acceptedMovesCounter,
            'improvingMovesCounter': sa.improvingMovesCounter,
            'acceptedMovesGain': sa.acceptedMovesGain
        }
        
        maxDays = max([data['patientInfo'][ind_p]['numFractions']+start for ind_p, start in enumerate(sa.solution.startDays)]) + 10

        data['daysQty'] = maxDays
        
        if sa.solution.checkIfLegal(sa.input):
            directorySol = f'solutions/FirstFit/monthly/{args.dataset_title}/{args.arrival_mean}_{args.time_windows_number}'
            if not os.path.exists(directorySol):
                os.makedirs(directorySol)
            with open(f'{directorySol}/{instance}.pk', 'wb') as handle:
                pickle.dump(sol_to_store, handle, protocol=pickle.HIGHEST_PROTOCOL)

            directoryInst = f'instances/monthly/{args.dataset_title}/{args.arrival_mean}_{args.time_windows_number}'
            if not os.path.exists(directoryInst):
                os.makedirs(directoryInst)
            with open(f"{directoryInst}/{instance}.pk", "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print("saved instance", instance)
        else:
            print("not legal solution instance:", instance)
