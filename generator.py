import pickle
import os
import argparse
import random
import numpy as np
import json
import glob

from data_structure.Input import Input
from solvers.heuristics.heuristics import heuristicFirstFit


# ---------------------------------------------------------------------------
# CLI – only two optional arguments: which config file(s) to run and
# an override for the global generator_parameter.json path.
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Generate RTSP benchmark instances from JSON config files in gen_input_params/.'
)
parser.add_argument(
    '-p', '--params',
    dest='params_pattern',
    default=None,
    metavar='PATTERN',
    help=(
        'Glob pattern (relative to gen_input_params/) to select which config '
        'files to run. Examples: "dataset_1_7_2.json", "dataset_6_*.json", '
        '"*.json" (default: all files in gen_input_params/).'
    )
)
parser.add_argument(
    '--gen-params',
    dest='gen_params_file',
    default='generator_parameter.json',
    metavar='FILE',
    help='Path to the global generator_parameter.json file (default: generator_parameter.json).'
)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Load global generation parameters
# ---------------------------------------------------------------------------
with open(args.gen_params_file) as f:
    parameters = json.load(f)


# ---------------------------------------------------------------------------
# Collect config files to process
# ---------------------------------------------------------------------------
GEN_INPUT_DIR = 'gen_input_params'

if args.params_pattern:
    pattern = os.path.join(GEN_INPUT_DIR, args.params_pattern)
    config_files = sorted(glob.glob(pattern))
    if not config_files:
        raise FileNotFoundError(
            f"No config files found matching pattern '{pattern}'."
        )
else:
    config_files = sorted(glob.glob(os.path.join(GEN_INPUT_DIR, '*.json')))
    if not config_files:
        raise FileNotFoundError(
            f"No JSON config files found in '{GEN_INPUT_DIR}/'."
        )

print(f"Found {len(config_files)} config file(s) to process:")
for cf in config_files:
    print(f"  {cf}")
print()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
time_horizon = 150
daysPatientsArrival = 25
weights = [10, 3, 1]
dayTarget = [2, 10, 20]


def get_random_tw_pref(num_tw):
    if num_tw < 2:
        return num_tw
    if random.uniform(0, 1) <= 0.8:
        return np.random.choice(num_tw, p=[1 / num_tw for _ in range(num_tw)])
    else:
        return None


# ---------------------------------------------------------------------------
# Main generation loop – one run per config file
# ---------------------------------------------------------------------------
for config_path in config_files:
    print(f"=== Processing config: {config_path} ===")

    with open(config_path) as f:
        cfg = json.load(f)

    dataset_title                 = cfg['dataset_title']
    instances_number              = cfg['instances_number']
    seed                          = cfg['seed']
    machines_number               = cfg['machines_number']
    time_windows_number           = cfg['time_windows_number']
    capacities                    = cfg['capacities']
    initial_occupation_percentage = cfg['initial_occupation_percentage']
    occupation_decay_velocity     = cfg['occupation_decay_velocity']
    arrival_mean                  = cfg['arrival_mean']
    priorities_percentage         = cfg['priorities_percentage']
    min_beam_matched_sets         = cfg.get('min_beam_matched_sets', 0)
    min_partial_beam_matched_sets = cfg.get('min_partial_beam_matched_sets', 0)

    random.seed(seed)
    np.random.seed(seed)

    # Protocol machine eligibility
    protocolMachineEligibility = []
    for protocol in parameters['patients']['protocols_info']:
        m_elig = np.zeros(machines_number)
        preferredLen = (
            np.random.randint(1, round(machines_number * 0.9))
            if round(machines_number * 0.9) > 1
            else 1
        )
        m_temp = np.random.choice(machines_number, preferredLen, replace=False)
        atLeastOnePreferred = False
        for m in m_temp:
            if atLeastOnePreferred and 0.6 < np.random.rand():
                m_elig[int(m)] = 1
            else:
                m_elig[int(m)] = 2
                atLeastOnePreferred = True
        protocolMachineEligibility.append(m_elig)

    # Beam matching
    partialSets = min_partial_beam_matched_sets
    machineBeamMatching = np.zeros((machines_number, machines_number))
    for ind in range(machines_number):
        machineBeamMatching[ind][ind] = 2

    m_remaining = np.arange(machines_number)
    completeSets = 0
    for _ in range(partialSets):
        complete = False
        setLen = (
            np.random.randint(2, machines_number // partialSets)
            if machines_number // partialSets > 2
            else 2
        )
        m_temp = np.random.choice(m_remaining, setLen, replace=False)
        for ind_m, m in enumerate(m_temp):
            m_remaining = np.delete(m_remaining, np.where(m_remaining == m))
            for m2 in m_temp[ind_m + 1:]:
                if m != m2:
                    if completeSets < min_beam_matched_sets:
                        complete = True
                        machineBeamMatching[m][m2] = 2
                        machineBeamMatching[m2][m] = 2
                    else:
                        machineBeamMatching[m][m2] = 1
                        machineBeamMatching[m2][m] = 1
        if complete:
            completeSets += 1

    print(f"  completeSets: {completeSets}")

    for instance in range(instances_number):

        data = {
            'machineEligibility': [],
            'machineBeamMatching': machineBeamMatching,
            'patientInfo': [],
            'machinesCapacity': [[] for _ in range(time_horizon)],
            'timeWindowsQty': time_windows_number,
            'patientsQty': None,
            'machinesQty': machines_number,
            'machinesMaxCapacity': capacities,
            'patientsGroupedByProtocol': None,
            'daysQty': time_horizon,
        }

        orderedProtocol = list(range(len(parameters['patients']['protocols_info'])))
        orderedProtocol.sort(
            key=lambda protocolId: parameters['patients']['protocols_info'][str(protocolId)]['priority']
        )
        data['patientsGroupedByProtocol'] = {i: [] for i in orderedProtocol}

        for m in range(machines_number):
            for tw in range(time_windows_number):
                ind = m * time_windows_number + tw
                beta_0 = np.log(initial_occupation_percentage[ind] * capacities[ind])
                beta_1 = (
                    occupation_decay_velocity[ind] * parameters['machines']['beta_1_min']
                    + (1 - occupation_decay_velocity[ind]) * parameters['machines']['beta_1_max']
                )
                beta_2 = parameters['machines']['beta_2_mean']
                for d in range(time_horizon):
                    mu_d = np.exp(beta_0 + beta_1 * d + beta_2 * (d ** 2))
                    k_d = (
                        parameters['machines']['k_d_median'][d]
                        if len(parameters['machines']['k_d_median']) > abs(d)
                        else 1000
                    )
                    if k_d != 1000:
                        occ_d = np.random.negative_binomial(
                            parameters['machines']['k_d_median'][d],
                            parameters['machines']['k_d_median'][d]
                            / (parameters['machines']['k_d_median'][d] + mu_d),
                        )
                    else:
                        occ_d = np.random.poisson(mu_d)
                    occ_d = min(capacities[ind], occ_d)
                    data['machinesCapacity'][d].append(capacities[ind] - occ_d)

        patientId = 0
        for d in range(daysPatientsArrival):
            patients_num = np.random.poisson(arrival_mean)
            for _ in range(patients_num):
                priority = np.random.choice(
                    np.arange(1, len(priorities_percentage) + 1),
                    p=priorities_percentage,
                )
                protocol = str(
                    np.random.choice(
                        np.array(
                            [int(el) for el in parameters['patients']['protocols_probabilities'][priority - 1].keys()]
                        ),
                        p=[
                            float(el) / sum(
                                float(v) for v in parameters['patients']['protocols_probabilities'][priority - 1].values()
                            )
                            for el in parameters['patients']['protocols_probabilities'][priority - 1].values()
                        ],
                    )
                )

                data['machineEligibility'].append(protocolMachineEligibility[int(protocol)])
                data['patientsGroupedByProtocol'][int(protocol)].append(patientId)
                data['patientInfo'].append({
                    'cost': weights[priority - 1],
                    'dMin': d,
                    'dTarget': d + dayTarget[priority - 1],
                    'twPref': get_random_tw_pref(time_windows_number),
                    'numFractions': parameters['patients']['protocols_info'][protocol]['numFractions'],
                    'allowedDays': [
                        int(day + 1 in parameters['patients']['protocols_info'][protocol]['allowed_start_weekdays_protocol'])
                        for day in range(time_horizon)
                    ],
                    'priority': priority,
                    'fractionsDuration': (
                        [parameters['patients']['protocols_info'][protocol]['duration_first']]
                        + [
                            parameters['patients']['protocols_info'][protocol]['duration_others']
                            for _ in range(parameters['patients']['protocols_info'][protocol]['numFractions'] - 1)
                        ]
                    ),
                })
                patientId += 1

        data['patientsQty'] = len(data['patientInfo'])

        for protocolId in orderedProtocol:
            data['patientsGroupedByProtocol'][protocolId].sort(
                key=lambda pid: data['patientInfo'][pid]['dTarget']
            )

        inputData = Input('', data=data)

        for obj in range(1, 2):
            objWeights = [w for w in parameters['weightsAlpha'][str(obj)].values()]
            solution, _, timeElapsed = heuristicFirstFit(inputData)

            sol_to_store = {
                'patientAppointments': solution.patientAppointments,
                'startDays': solution.startDays,
            }

            maxDays = (
                max(
                    data['patientInfo'][ind_p]['numFractions'] + start
                    for ind_p, start in enumerate(solution.startDays)
                )
                + 10
            )
            data['daysQty'] = maxDays

            if solution.checkIfLegal(inputData):
                directorySol = f'solutions/FirstFit/monthly/{dataset_title}/{arrival_mean}_{time_windows_number}'
                os.makedirs(directorySol, exist_ok=True)
                with open(f'{directorySol}/{instance}.pk', 'wb') as handle:
                    pickle.dump(sol_to_store, handle, protocol=pickle.HIGHEST_PROTOCOL)

                directoryInst = f'instances/monthly/{dataset_title}/{arrival_mean}_{time_windows_number}'
                os.makedirs(directoryInst, exist_ok=True)
                with open(f'{directoryInst}/{instance}.pk', 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

                print(f"  saved instance {instance}")
            else:
                print(f"  NOT legal solution for instance {instance}")

    print()
