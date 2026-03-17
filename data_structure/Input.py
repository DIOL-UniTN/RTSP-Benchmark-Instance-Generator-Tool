import pickle as pk
import numpy as np

class Input:
    def __init__(self, fileName, patientsIdList = None, data = None):
        if data == None:
            with open(fileName, "rb") as input_file:
                data = pk.load(input_file)

        self.machineEligibility = data['machineEligibility']
        self.machineBeamMatching = data['machineBeamMatching']
        self.patientInfo = data['patientInfo']
        '''
        patientInfo: {
            'cost': cost for delayed treatment start,
            'dMin': minimum day for starting treatment,
            'dTarget': deadline day for starting treatment,
            'twPref': preferred time window of the patient (if any),
            'numFractions': number of treatment fractions to undergo,
            'allowedDays': array of pseudoboolean values for allowed days,
            'priority': priority of the patient,
            'fractionsDuration': array long as number of fractions, containing duration of each
        }
        '''
        self.machinesCapacity = np.array(data['machinesCapacity'])
        self.timeWindowsQty = data['timeWindowsQty']
        self.patientsQty = data['patientsQty']
        self.machinesQty = data['machinesQty']
        self.machinesMaxCapacity = data['machinesMaxCapacity']
        self.patientsGroupedByProtocol = data['patientsGroupedByProtocol']
        self.daysQty = data['daysQty']
        if patientsIdList != None:
            self.patientsIdList = patientsIdList
        else:
            self.patientsIdList = [el for el in range(data['patientsQty'])]