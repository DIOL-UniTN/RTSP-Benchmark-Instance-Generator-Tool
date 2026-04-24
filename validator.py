from data_structure.Solution import Solution
from data_structure.Input import Input


def main(inputFile, solutionFile):
    inputData = Input(inputFile)
    sol = Solution(inputData, solutionFile, None)
    valid = sol.checkIfLegal(inputData)
    if valid:
        print(f"Your solution for input file {inputFile} is FEASIBLE")
    else:
        print(f"Unfortunately your solution for input file {inputFile} is NOT feasible")
