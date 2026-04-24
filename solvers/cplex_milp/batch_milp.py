import argparse
import cplex
from data_structure.Input import Input
from data_structure.Solution import Solution
from data_structure.Fitness import Fitness
import time

def solveBatch(patientsDaily, alphas, inputData, windowsSet, allBeamMatchedMachines, completelyBeamMatchedMachines, costFunctions, sol_to_store, time_limit, seed):
    # Create a CPLEX model instance
    problem = cplex.Cplex()
    problem.set_log_stream(None)
    problem.set_error_stream(None)
    problem.set_warning_stream(None)
    problem.set_results_stream(None)
    problem.parameters.randomseed = seed

    # Set optimization sense (minimization)
    # problem.set_problem_type(cplex.Cplex.problem_type.MIQCP)
    # problem.objective.set_sense(cplex._internal._constants.CPX_MIN)
    problem.objective.set_sense(problem.objective.sense.minimize)
    problem.parameters.timelimit.set(time_limit)

    x = {}
    t = {}
    q = {}
    z = {}
    u = {}
    y = {}
    v = {}
    s = {}
    f_1 = {}
    f_2 = {}
    f_3 = {}
    f_4 = {}
    f_5 = {}
    f_6 = {}

    alpha_1 = alphas['alpha_1']
    alpha_2 = alphas['alpha_2']
    alpha_3 = alphas['alpha_3']
    alpha_4 = alphas['alpha_4']
    alpha_5 = alphas['alpha_5']
    alpha_6 = alphas['alpha_6']
    
    var_index = 0

    variable_names = []
    objective_coeffs = []
    lower_bounds = []
    upper_bounds = []
    var_types = []
    for p in patientsDaily:
        for m in range(inputData.machinesQty):
            for d in range(inputData.daysQty):
                for w in windowsSet:
                    x[(p, m, d, w)] = var_index
                    var_index += 1
                    variable_names.append(f"x_{p}_{m}_{d}_{w}")
                    objective_coeffs.append(0)
                    lower_bounds.append(0)
                    upper_bounds.append(1)
                    var_types.append(problem.variables.type.binary)

                    t[(p, m, d, w)] = var_index
                    var_index += 1
                    variable_names.append(f"t_{p}_{m}_{d}_{w}")
                    objective_coeffs.append(0)
                    lower_bounds.append(0)
                    upper_bounds.append(1)
                    var_types.append(problem.variables.type.binary)

                for f in range(inputData.patientInfo[p]['numFractions']):
                    q[(p, m, d, f)] = var_index
                    var_index += 1
                    variable_names.append(f"q_{p}_{m}_{d}_{f}")
                    objective_coeffs.append(0)
                    lower_bounds.append(0)
                    upper_bounds.append(1)
                    var_types.append(problem.variables.type.binary)

    for p in patientsDaily:
        f_1[p] = var_index
        var_index += 1
        variable_names.append(f"f_1_{p}")
        objective_coeffs.append(alpha_1)
        lower_bounds.append(0)
        upper_bounds.append(cplex.infinity)
        var_types.append(problem.variables.type.integer)

        f_2[p] = var_index
        var_index += 1
        variable_names.append(f"f_2_{p}")
        objective_coeffs.append(alpha_2)
        lower_bounds.append(0)
        upper_bounds.append(cplex.infinity)
        var_types.append(problem.variables.type.integer)

        f_3[p] = var_index
        var_index += 1
        variable_names.append(f"f_3_{p}")
        objective_coeffs.append(alpha_3)
        lower_bounds.append(0)
        upper_bounds.append(cplex.infinity)
        var_types.append(problem.variables.type.integer)

        f_4[p] = var_index
        var_index += 1
        variable_names.append(f"f_4_{p}")
        objective_coeffs.append(alpha_4)
        lower_bounds.append(0)
        upper_bounds.append(cplex.infinity)
        var_types.append(problem.variables.type.integer)

        f_5[p] = var_index
        var_index += 1
        variable_names.append(f"f_5_{p}")
        objective_coeffs.append(alpha_5)
        lower_bounds.append(0)
        upper_bounds.append(cplex.infinity)
        var_types.append(problem.variables.type.integer)

        f_6[p] = var_index
        var_index += 1
        variable_names.append(f"f_6_{p}")
        objective_coeffs.append(alpha_6)
        lower_bounds.append(0)
        upper_bounds.append(cplex.infinity)
        var_types.append(problem.variables.type.integer)

        for d in range(inputData.daysQty):
            z[(p, d)] = var_index
            var_index += 1
            variable_names.append(f"z_{p}_{d}")
            objective_coeffs.append(0)
            lower_bounds.append(0)
            upper_bounds.append(1)
            var_types.append(problem.variables.type.binary)

            u[(p, d)] = var_index
            var_index += 1
            variable_names.append(f"u_{p}_{d}")
            objective_coeffs.append(0)
            lower_bounds.append(0)
            upper_bounds.append(windowsSet[-1])
            var_types.append(problem.variables.type.integer)

            for w in windowsSet:
                y[(p, d, w)] = var_index
                var_index += 1
                variable_names.append(f"y_{p}_{d}_{w}")
                objective_coeffs.append(0)
                lower_bounds.append(0)
                upper_bounds.append(1)
                var_types.append(problem.variables.type.binary)
        for f in range(inputData.patientInfo[p]['numFractions'])[:-1]:
            v[(p, f)] = var_index
            var_index += 1
            variable_names.append(f"v_{p}_{f}")
            objective_coeffs.append(0)
            lower_bounds.append(0)
            upper_bounds.append(1)
            var_types.append(problem.variables.type.binary)

            s[(p, f)] = var_index
            var_index += 1
            variable_names.append(f"s_{p}_{f}")
            objective_coeffs.append(0)
            lower_bounds.append(0)
            upper_bounds.append(1)
            var_types.append(problem.variables.type.binary)

        v[(p, range(inputData.patientInfo[p]['numFractions'])[-1])] = var_index
        var_index += 1
        variable_names.append(f"v_{p}_{range(inputData.patientInfo[p]['numFractions'])[-1]}")
        objective_coeffs.append(0)
        lower_bounds.append(0)
        upper_bounds.append(1)
        var_types.append(problem.variables.type.binary)


    # Constraints for objective function
    constraints = []
    rightHandSide = []
    constraint_senses = []
    const_names = []
    for p in patientsDaily:
        constraints.append([
            [q[(p, m, d, 0)] for d in range(inputData.daysQty) for m in range(inputData.machinesQty) if d >= inputData.patientInfo[p]['dMin']] + [f_1[p]],
            [(d - inputData.patientInfo[p]['dMin']) * inputData.patientInfo[p]['cost'] for d in range(inputData.daysQty) for m in range(inputData.machinesQty) if d >= inputData.patientInfo[p]['dMin']] + [-1]
        ])
        rightHandSide.append(0)
        constraint_senses.append("E")
        const_names.append(f'f_1_{p}')
        # f_1 
        # model.add_constr(sum(q[(p, m, d, 0)] * (d - inputData.patientInfo[p]['dMin']) * inputData.patientInfo[p]['cost'] for d in range(inputData.daysQty) for m in range(inputData.machinesQty) if d >= inputData.patientInfo[p]['dMin']) == f_1[p])
        
        constraints.append([
            [q[(p, m, d, 0)] for d in range(inputData.daysQty) for m in range(inputData.machinesQty) if d >= inputData.patientInfo[p]['dTarget']] + [f_2[p]],
            [(d - inputData.patientInfo[p]['dTarget']) * inputData.patientInfo[p]['cost'] for d in range(inputData.daysQty) for m in range(inputData.machinesQty) if d >= inputData.patientInfo[p]['dTarget']] + [-1]
        ])
        rightHandSide.append(0)
        constraint_senses.append("E")
        const_names.append(f'f_2_{p}')
        # f_2 
        # model.add_constr(sum(q[(p, m, d, 0)] * (d - inputData.patientInfo[p]['dTarget']) * inputData.patientInfo[p]['cost'] for d in range(inputData.daysQty) for m in range(inputData.machinesQty) if d >= inputData.patientInfo[p]['dTarget']) == f_2[p])
        
        constraints.append([
            [z[(p, d)] for d in range(inputData.daysQty)] + [f_3[p]],
            [1 for d in range(inputData.daysQty)] + [-1]
        ])
        rightHandSide.append(0)
        constraint_senses.append("E")
        const_names.append(f'f_3_{p}')
        # f_3 
        # model.add_constr(sum(z[(p, d)] for d in range(inputData.daysQty)) == f_3[p])

        constraints.append([
            [u[(p, d)] for d in range(inputData.daysQty)] + [f_4[p]],
            [1 for d in range(inputData.daysQty)] + [-1]
        ])
        rightHandSide.append(0)
        constraint_senses.append("E")
        const_names.append(f'f_4_{p}')
        # f_4
        # model.add_constr(sum(u[(p, d)] for d in range(inputData.daysQty)) == f_4[p])

        constraints.append([
            [v[(p, f)] for f in range(inputData.patientInfo[p]['numFractions'])] + [f_5[p]],
            [1 for f in range(inputData.patientInfo[p]['numFractions'])] + [-1]
        ])
        rightHandSide.append(0)
        constraint_senses.append("E")
        const_names.append(f'f_5_{p}')
        # f_5
        # model.add_constr(sum(v[(p, f)] for f in range(inputData.patientInfo[p]['numFractions'])) == f_5[p])

        constraints.append([
            [s[(p, f)] for f in range(inputData.patientInfo[p]['numFractions'])[:-1]] + [f_6[p]],
            [1 for f in range(inputData.patientInfo[p]['numFractions'])[:-1]] + [-1]
        ])
        rightHandSide.append(0)
        constraint_senses.append("E")
        const_names.append(f'f_6_{p}')
        # f_6 "Number of times the machine has been switched to a partialle matched one between fractions"
        # model.add_constr(sum(s[(p, f)] for f in range(inputData.patientInfo[p]['numFractions'])[:-1]) == f_6[p])

    # Constraints
    days_list = list(range(inputData.daysQty))

    for p in patientsDaily:
        fractions = inputData.patientInfo[p]['numFractions']
        for ind_mm, matchedMachines in enumerate(allBeamMatchedMachines):
                for f in range(fractions-1):
                    prev_d = days_list[0]
                    for d in days_list[1:]:
                        constraints.append([
                            [q[(p, m, prev_d, f)] for m in matchedMachines] + [q[(p, m, d, f+1)] for m in matchedMachines],
                            [1 for m in matchedMachines] + [-1 for m in matchedMachines]
                        ])
                        rightHandSide.append(0)
                        constraint_senses.append("E")
                        const_names.append(f'c_2_{p}_{ind_mm}_{f}_{d}')
                        # constr 2 "Fractions must be consecutive by days and only on all beam matched machines"
                        # model.add_constr(
                        #     sum(q[(p, m, prev_d, f)] for m in matchedMachines) ==
                        #     sum(q[(p, m, d, f+1)] for m in matchedMachines)
                        # )
                        prev_d = d

        for f in range(inputData.patientInfo[p]['numFractions']):
            constraints.append([
                [q[(p, m, d, f)] for m in range(inputData.machinesQty) for d in range(inputData.daysQty)],
                [1 for m in range(inputData.machinesQty) for d in range(inputData.daysQty)]
            ])
            rightHandSide.append(1)
            constraint_senses.append("E")
            const_names.append(f'c_3_{p}_{f}')
            # constr 3 "Fraction f must be scheduled at least and at most once"
            # model.add_constr(
            #     sum(q[(p, m, d, f)] for m in range(inputData.machinesQty) for d in range(inputData.daysQty)) == 1
            # )

        for m in range(inputData.machinesQty):
            for d in range(inputData.daysQty):
                constraints.append([
                    [q[(p, m, d, 0)]] + [t[(p, m, d, w)] for w in windowsSet],
                    [1] + [-1 for w in windowsSet]
                ])
                rightHandSide.append(0)
                constraint_senses.append("E")
                const_names.append(f'c_4_{p}_{m}_{d}')
                # constr 4 "The first fraction for patient p is scheduled on machine m on day d, in any window,"
                # model.add_constr(
                #     q[(p, m, d, 0)] == 
                #     sum(t[(p, m, d, w)] for w in windowsSet)
                # )
                for w in windowsSet:
                    constraints.append([
                        [t[(p, m, d, w)]] + [x[(p, m, d, w)]],
                        [1] + [-1]
                    ])
                    rightHandSide.append(0)
                    constraint_senses.append("L")
                    const_names.append(f'c_5_{p}_{m}_{d}_{w}')
                    # constr 5 "Correct window w for the first fraction."
                    # model.add_constr(
                    #     t[(p, m, d, w)] <=
                    #     x[(p, m, d, w)]
                    # )
                if (d > inputData.daysQty - inputData.patientInfo[p]['numFractions'] or d < inputData.patientInfo[p]['dMin'] or not inputData.patientInfo[p]['allowedDays'][d]): # Assumo d_min = 0
                    constraints.append([
                        [q[(p, m, d, 0)]],
                        [1]
                    ])
                    rightHandSide.append(0)
                    constraint_senses.append("E")
                    const_names.append(f'c_6_{p}_{m}_{d}')
                    # constr 6
                    # model.add_constr(
                    #     q[(p, m, d, 0)] == 0
                    # )
                for f in range(inputData.patientInfo[p]['numFractions']):
                    if (inputData.machineEligibility[p][m] == 0 or (f < d and (d - f) > (inputData.daysQty - inputData.patientInfo[p]['numFractions']))): 
                        constraints.append([
                            [q[(p, m, d, f)]],
                            [1]
                        ])
                        rightHandSide.append(0)
                        constraint_senses.append("E")
                        const_names.append(f'c_7_{p}_{m}_{d}_{f}')
                        # constr 7
                        # model.add_constr(
                        #     q[(p, m, d, f)] == 0
                        # )
                constraints.append([
                    [x[(p, m, d, w)] for w in windowsSet] + [q[(p, m, d, f)] for f in range(inputData.patientInfo[p]['numFractions'])],
                    [1 for w in windowsSet] + [-1 for f in range(inputData.patientInfo[p]['numFractions'])]
                ])
                rightHandSide.append(0)
                constraint_senses.append("E")
                const_names.append(f'c_8_{p}_{m}_{d}')
                # constr 8
                # model.add_constr(
                #     sum(x[(p, m, d, w)] for w in windowsSet) ==
                #     sum(q[(p, m, d, f)] for f in range(inputData.patientInfo[p]['numFractions']))
                # )
        
        for ind_d in range(len(days_list)):
            d = days_list[ind_d]
            d_next = days_list[ind_d+1] if ind_d+1 < len(days_list) else None
            constraints.append([
                [y[(p, d, w)] for w in windowsSet],
                [1 for w in windowsSet]
            ])
            rightHandSide.append(1)
            constraint_senses.append("E")
            const_names.append(f'c_12_{p}_{d}')
            # constr 12
            # model.add_constr(
            #     sum(y[(p, d, w)] for w in windowsSet) == 1
            # )
            for w in windowsSet:
                constraints.append([
                    [y[(p, d, w)]] + [x[(p, m, d, w)] for m in range(inputData.machinesQty)],
                    [1] + [-1 for m in range(inputData.machinesQty)]
                ])
                rightHandSide.append(0)
                constraint_senses.append("G")
                const_names.append(f'c_11_{p}_{d}_{w}')
                # constr 11
                # model.add_constr(
                #     y[(p, d, w)] >=
                #     sum(x[(p, m, d, w)] for m in range(inputData.machinesQty))
                # )
                if d_next != None:
                    constraints.append([
                        [z[(p, d)]] + [y[(p, d, w)], y[(p, d_next, w)]],
                        [1] + [-1, 1]
                    ])
                    rightHandSide.append(0)
                    constraint_senses.append("G")
                    const_names.append(f'c_13_{p}_{d}_{w}')
                    # constr 13
                    # model.add_constr(
                    #     z[(p, d)] >=
                    #     y[(p, d, w)] - y[(p, d_next.id, w)]
                    # )
                    constraints.append([
                        [z[(p, d)]] + [y[(p, d_next, w)], y[(p, d, w)]],
                        [1] + [-1, 1]
                    ])
                    rightHandSide.append(0)
                    constraint_senses.append("G")
                    const_names.append(f'c_14_{p}_{d}_{w}')
                    # constr 14
                    # model.add_constr(
                    #     z[(p, d)] >=
                    #     y[(p, d_next.id, w)] - y[(p, d, w)]
                    # )
            if inputData.patientInfo[p]['twPref'] != None:
                constraints.append([
                    [u[(p, d)]] + [x[(p, m, d, w)] for m in range(inputData.machinesQty) for w in windowsSet],
                    [1] + [-1 * abs(w - inputData.patientInfo[p]['twPref']) for m in range(inputData.machinesQty) for w in windowsSet]
                ])
                rightHandSide.append(0)
                constraint_senses.append("E")
                const_names.append(f'c_15_{p}_{d}')
                # constr 15
                # model.add_constr(
                #     u[(p, d)] == 
                #     sum(x[(p, m, d, w)] * abs(w - inputData.patientInfo[p]['twPref']) for m in range(inputData.machinesQty) for w in windowsSet)
                # )
            else:
                constraints.append([
                    [u[(p, d)]],
                    [1]
                ])
                rightHandSide.append(0)
                constraint_senses.append("E")
                const_names.append(f'c_16_{p}_{d}')
                # constr 16
                # model.add_constr(
                #     u[(p, d)] == 0
                # )
        for f in range(inputData.patientInfo[p]['numFractions']):
            for ind_cbm, c_bm in enumerate(completelyBeamMatchedMachines):
                if f+1 < inputData.patientInfo[p]['numFractions']:
                    constraints.append([
                        [s[(p, f)]] + [q[(p, m, d, f)] for d in range(inputData.daysQty) for m in c_bm] + [q[(p, m, d, f+1)] for d in range(inputData.daysQty) for m in c_bm],
                        [1] + [-1 for d in range(inputData.daysQty) for m in c_bm] + [1 for d in range(inputData.daysQty) for m in c_bm]
                    ])
                    rightHandSide.append(0)
                    constraint_senses.append("G")
                    const_names.append(f'c_17_{p}_{d}_{f}_{ind_cbm}')
                    # constr 17
                    # model.add_constr(
                    #     s[(p, f)] >=
                    #     sum(q[(p, m, d, f)] - q[(p, m, d, f+1)] for d in range(inputData.daysQty) for m in c_bm)
                    # )
            constraints.append([
                [v[(p, f)]] + [q[(p, m, d, f)] for d in range(inputData.daysQty) for m in range(inputData.machinesQty) if inputData.machineEligibility[p][m] != 2], 
                [1] + [-1 for d in range(inputData.daysQty) for m in range(inputData.machinesQty) if inputData.machineEligibility[p][m] != 2]
            ])
            rightHandSide.append(0)
            constraint_senses.append("E")
            const_names.append(f'c_18_{p}_{d}_{f}')
            # constr 18
            # model.add_constr(
            #     v[(p, f)] ==
            #     sum(q[(p, m, d, f)] for d in range(inputData.daysQty) for m in range(inputData.machinesQty) if inputData.machineEligibility[p][m] != 2)
            # )

    
    for m in range(inputData.machinesQty):
        for d in range(inputData.daysQty):
            for w in windowsSet:
                available = float(inputData.machinesCapacity[d][(m*inputData.timeWindowsQty)+w])
                firstFractionDur = float(inputData.patientInfo[p]['fractionsDuration'][0])
                fractionDur = float(inputData.patientInfo[p]['fractionsDuration'][1]) if len(inputData.patientInfo[p]['fractionsDuration']) > 1 else float(inputData.patientInfo[p]['fractionsDuration'][0])
                # available = data.timeWindows[f"{d}_{m}_{w}"].capacity - data.timeWindows[f"{d}_{m}_{w}"].occupation
                # available = available if available >= 0 else 0

                constraints.append([
                    [x[(p, m, d, w)] for p in patientsDaily] + 
                    [t[(p, m, d, w)] for p in patientsDaily],

                    [float(inputData.patientInfo[p]['fractionsDuration'][1]) if len(inputData.patientInfo[p]['fractionsDuration']) > 1 else float(inputData.patientInfo[p]['fractionsDuration'][0]) for p in patientsDaily] + 
                    [(float(inputData.patientInfo[p]['fractionsDuration'][0] - (inputData.patientInfo[p]['fractionsDuration'][1] if len(inputData.patientInfo[p]['fractionsDuration']) > 1 else inputData.patientInfo[p]['fractionsDuration'][0]))) for p in patientsDaily]
                ])
                rightHandSide.append(available)
                constraint_senses.append("L")
                const_names.append(f'c_9_{m}_{d}_{w}')
                # constr 9
                # model.add_constr(
                #     sum([(x[(p, m, d, w)] - t[(p, m, d, w)]) * inputData.patientInfo[p]['fractionsDuration'][1] + t[(p, m, d, w)] * inputData.patientInfo[p]['fractionsDuration'][0] for p in patientsDaily]) <= 
                #     available
                # )

    temp = dict(sorted(inputData.patientsGroupedByProtocol.items()))
    for ind_treat, patients_protocol in temp.items(): #data.treatments.values():
        filtered_patients_protocol = [el for el in patients_protocol if el in patientsDaily]
        if filtered_patients_protocol != []: 
        # patients_protocol = [p for p in patientsDaily if p.treatmentProtocol.id == treatment.id]
        # patients_protocol.sort(key=lambda x: x.targetDay)
            for ind_p in range(len(filtered_patients_protocol)-1):
                constraints.append([
                    [q[(filtered_patients_protocol[ind_p], m, d, 0)] for m in range(inputData.machinesQty) for d in range(inputData.daysQty)] + 
                    [q[(filtered_patients_protocol[ind_p+1], m, d, 0)] for m in range(inputData.machinesQty) for d in range(inputData.daysQty)],
                    [d for m in range(inputData.machinesQty) for d in range(inputData.daysQty)] + [-d for m in range(inputData.machinesQty) for d in range(inputData.daysQty)]
                ])
                rightHandSide.append(0)
                constraint_senses.append("L")
                const_names.append(f'c_10_{ind_treat}_{ind_p}')
                # constr 10
                # model.add_constr(
                #     sum(d*sum(q[(patients_protocol[ind_p].id, m, d, 0)] for m in range(inputData.machinesQty)) for d in range(inputData.daysQty)) <=
                #     sum(d*sum(q[(patients_protocol[ind_p+1].id, m, d, 0)] for m in range(inputData.machinesQty)) for d in range(inputData.daysQty))
                # )
    
    problem.variables.add(obj=objective_coeffs,
                        lb=lower_bounds,
                        ub=upper_bounds,
                        names=variable_names,
                        types=var_types)
    
    # Add constant to the objective function
    problem.objective.set_offset(1)  # This adds "+1" to the objective function

    problem.linear_constraints.add(lin_expr=constraints,
                               senses=constraint_senses,
                               rhs=rightHandSide,
                               names=const_names)
    
    # Capture the time before solving
    start_time = problem.get_time()
    
    # Solve the problem
    problem.solve()

    # Capture the time after solving
    end_time = problem.get_time()

    # Compute the elapsed time
    solver_time = end_time - start_time

    status = problem.solution.get_status()
    status_str = problem.solution.get_status_string()
    print(f"Solution status: {status} ({status_str})")

    if status in [1, 101, 102, 105, 107, 109, 111, 113, 116]:
        solution_values = problem.solution.get_values()
        for p in patientsDaily:
            for m in range(inputData.machinesQty):
                for d in range(inputData.daysQty):
                    for w in windowsSet:
                        if solution_values[x[(p, m, d, w)]] == 1:
                            for f in range(inputData.patientInfo[p]['numFractions']):
                                if solution_values[q[(p, m, d, f)]] == 1:
                                    inputData.machinesCapacity[d][(m*inputData.timeWindowsQty)+w] -= inputData.patientInfo[p]['fractionsDuration'][f]
                                    #data.timeWindows[f"{d}_{m}_{w}"].updateOccupation(inputData.patientInfo[p]['fractionsDuration'][f])
                                    sol_to_store[(p, f'M{m+1}', d, w)] = f
                                    assignment = None
                                    startDays = None

            costFunctions['f_1'][p] = solution_values[f_1[p]]
            costFunctions['f_2'][p] = solution_values[f_2[p]]
            costFunctions['f_3'][p] = solution_values[f_3[p]]
            costFunctions['f_4'][p] = solution_values[f_4[p]]
            costFunctions['f_5'][p] = solution_values[f_5[p]]
            costFunctions['f_6'][p] = solution_values[f_6[p]]
        
        print(f"f_1 sum: {sum(costFunctions['f_1'][p] for p in patientsDaily)}")
        print(f"f_2 sum: {sum(costFunctions['f_2'][p] for p in patientsDaily)}")
        print(f"f_3 sum: {sum(costFunctions['f_3'][p] for p in patientsDaily)}")
        print(f"f_4 sum: {sum(costFunctions['f_4'][p] for p in patientsDaily)}")
        print(f"f_5 sum: {sum(costFunctions['f_5'][p] for p in patientsDaily)}")
        print(f"f_6 sum: {sum(costFunctions['f_6'][p] for p in patientsDaily)}")
        print('---------------------')
        print('Objective: ', problem.solution.get_objective_value())
        print(f'Time: {solver_time:.4f}')
        print('---------------------')
    return inputData, costFunctions, assignment, startDays, status


def solve(args, alphas):
    costFunctions = {
        'f_1': {},
        'f_2': {},
        'f_3': {},
        'f_4': {},
        'f_5': {},
        'f_6': {}
    }
    inputData = Input(args.instance_file)
    patientsBatch = [[p_id for p_id, patient in enumerate(inputData.patientInfo) if patient['dMin'] == day] for day in range(inputData.daysQty)]
    patientsBatch = list(filter(None, patientsBatch))
    
    completelyBeamMatchedMachines = [[machineId for machineId in range(inputData.machinesQty) if inputData.machineBeamMatching[machineId][m2Id] == 2] for m2Id in range(inputData.machinesQty)]
    temp_beamMatchedMachines = []
    for el in completelyBeamMatchedMachines:
        if el not in temp_beamMatchedMachines:
            temp_beamMatchedMachines.append(el)
    completelyBeamMatchedMachines = temp_beamMatchedMachines   
    completelyBeamMatchedMachines.sort(key=lambda x: len(x), reverse=True)
    
    allBeamMatchedMachines = [[machineId for machineId in range(inputData.machinesQty) if inputData.machineBeamMatching[machineId][m2Id] != 0] for m2Id in range(inputData.machinesQty)]
    temp_beamMatchedMachines = []
    for el in allBeamMatchedMachines:
        if el not in temp_beamMatchedMachines:
            temp_beamMatchedMachines.append(el)
    allBeamMatchedMachines = temp_beamMatchedMachines   
    allBeamMatchedMachines.sort(key=lambda x: len(x), reverse=True)

    patientsList = list(range(len(inputData.patientInfo)))

    windowSet = [el for el in range(inputData.timeWindowsQty)]

    sol_to_store = {}
    count = 0
    feasibility = True
    time_pieces = (args.time_limit * 3 / 4) / (len(patientsList) - len(patientsBatch[0]))
    start_time = time.time()

    assignment = [[[None, None] for k in range(len(inputData.patientInfo))] for i in range(inputData.daysQty)]
    startDays = [None for k in range(len(inputData.patientInfo))]

    for ind_pd, patientsDaily in enumerate(patientsBatch):
        if ind_pd == 0:
            time_limit = args.time_limit / 4
        else:
            time_limit = time_pieces * len(patientsDaily)
        print('time batch', time_limit)
        count += len(patientsDaily)
        print(f"{count} out of {len(patientsList)} patients")
        inputData, costFunctions, assignment, startDays, status = solveBatch(patientsDaily, alphas[args.obj], inputData, windowSet, allBeamMatchedMachines, completelyBeamMatchedMachines, costFunctions, assignment, startDays, time_limit, args.seed)
        if status not in [1, 101, 102, 105, 107, 109, 111, 113, 116]:
            feasibility = False
            break
        if (time.time() - start_time) > args.time_limit:
            feasibility = False
            break
    
    elapsed_time = ((time.time() - start_time))

    solution = Solution(inputData, None, {'patientAppointments': assignment, 'startDays': startDays}, False)

    if feasibility and solution.checkIfLegal(inputData):
        objWeights = [w for w in alphas[args.obj].values()]
        fitness = Fitness(solution, inputData, objWeights)
        f_obj = [row.sum() for row in fitness.objectiveMatrix.transpose()]

        sol_to_store = {
            'patientAppointments': solution.patientAppointments,
            'startDays': solution.startDays,
        }
        f_1_tot = sum([costFunctions['f_1'][p] for p in range(inputData.patientsQty)])
        f_2_tot = sum([costFunctions['f_2'][p] for p in range(inputData.patientsQty)])
        f_3_tot = sum([costFunctions['f_3'][p] for p in range(inputData.patientsQty)])
        f_4_tot = sum([costFunctions['f_4'][p] for p in range(inputData.patientsQty)])
        f_5_tot = sum([costFunctions['f_5'][p] for p in range(inputData.patientsQty)])
        f_6_tot = sum([costFunctions['f_6'][p] for p in range(inputData.patientsQty)])

        objective_val = (
            1 + 
            f_1_tot*alphas[args.obj]['alpha_1'] +
            f_2_tot*alphas[args.obj]['alpha_2'] +
            f_3_tot*alphas[args.obj]['alpha_3'] +
            f_4_tot*alphas[args.obj]['alpha_4'] +
            f_5_tot*alphas[args.obj]['alpha_5'] +
            f_6_tot*alphas[args.obj]['alpha_6']
        )

        if f_1_tot != f_obj[0]:
            raise Exception("f 1 not matching from cplex to data structure")
        if f_2_tot != f_obj[1]:
            raise Exception("f 2 not matching from cplex to data structure")
        if f_3_tot != f_obj[2]:
            raise Exception("f 3 not matching from cplex to data structure")
        if f_4_tot != f_obj[3]:
            raise Exception("f 4 not matching from cplex to data structure")
        if f_5_tot != f_obj[4]:
            raise Exception("f 5 not matching from cplex to data structure")
        if f_6_tot != f_obj[5]:
            raise Exception("f 6 not matching from cplex to data structure")
        if objective_val != fitness.objective:
            raise Exception("objective not matching from cplex to data structure")

        csvDict = {
            'obj': args.obj,
            'code': args.gen,
            'instance': args.instance,
            'time to optimum': elapsed_time,
            'time to feasible':  0,
            'solver status': 'Feasible',
            'objective': fitness.objective,
            'f1': f_obj[0],
            'f2': f_obj[1],
            'f3': f_obj[2],
            'f4': f_obj[3],
            'f5': f_obj[4],
            'f6': f_obj[5],
            'seed': args.seed 
        }
    else:
        sol_to_store = None
        csvDict = {
            'obj': args.obj,
            'code': args.gen,
            'instance': args.instance,
            'time to optimum': '---',
            'time to feasible': '---',
            'solver status': 'Not Feasible',
            'objective': '---',
            'f1': '---',
            'f2': '---',
            'f3': '---',
            'f4': '---',
            'f5': '---',
            'f6': '---',
            'seed': args.seed 
        }
    return {'pickle': sol_to_store, 'csv': csvDict}
