
from DataFrame import DataFrame

# -------------------------------------------------------------
# File:
# -----
#   EvaluationModule.py
# -------------------------------------------------------------
# Description:
# ------------
#   The EvaluationModule.py file contains the EvaluationModule
#   class. The EvaluationModule class has two purposes:
#      
#       1. Own the instance of a DataFrame class
#       2. Act as the source of truth for the parameters of the
#          simulation
#
# -------------------------------------------------------------
# Usage:
# ------
#   In order to experiment with how the Genetic Algorithm evolves
#   the neural networks described by the Mapping Operators in its
#   population, one can alter the global parameters in this file
#   between runs. There are many parameters in this implementation
#   that control how the algorithm runs. To expedite experimentation
#   it is easier to selectively change this source code and run
#   the algorithm again instead of entering custom parameters at
#   each run. Future work on this project might include some sort
#   of interface but for such small scale experiments there is
#   little need for one. There is a description for each parameter
#   below.
# -------------------------------------------------------------
# Notes:
# ------
#   Performance metrics are included here but have not been
#   implemented in the algorithm.
#
#   **The default values in each of the relevant files are overriden
#   by the values here.**
# -------------------------------------------------------------

class EvaluationModule :

    # Data Frame
    # ----------
    dataFrame = None

    # Best Mapping Operator
    #   ** This is the best Mapping Operator that is the result of training
    bestMappingOperator = None

    # Algorithm Parameters
    #   ** These parameters are the source of truth for the parameters throughout the algorithm
    # --------------------
    
    # GA Parameters
    populationSizeFactor = 5 # This value is multiplied by the cpu count to size the population
    maxGenerations = 1000000000 # The maximum number of generations the GA will run before quiting
    crossoverRate = .8  # The rate at which the members of the population are crossed over per iteration of the GA
    mutationRate = .1   # The rate at which the members of the population are mutated per iteration of the GA
    mutationLikelihood = .1 # The rate at which the weights of a member Mapping Operator are mutated
    biasMutationLikelihood = .1 # The rate at which the biases of the Mapping Operator are mutated
    topologicalMutationRate = 0 # The rate at which the depth of the Mapping Operator is mutated
    valueReplacementBias = .05 # The rate at which individual weights of the Mapping Operators are replaced versus adjusted
    mutationMagnitudeLow = .00000001 # The lowest possible adjustment value that can happen to a weight during mutation
    mutationMagnitudeHigh = .0001 # The highest possible adjustment value that can happen to a weight during mutation
    elitismWeight = 0 # The proportion of the population that will be saved as elite members and injected at the next generation

    # GA Algorithm Parameters
    selectionMethodIndicator = 1 # 0 = Roulette Wheel Selection, 1 = Tournament Selection :: The selection algorithm used in the GA
    rouletteWheelselectionBias = .08 # The bias towards fitter members in Roulette Wheel Selection. Lower values favor fitter members
    tournamentPopulationProportion = .3 # The proportion of the population that is to compete in Tournament Selection
    
    # Data Frame Parameters
    stimulusProductPairCount = 2 # The number of Stimulus-Product Pairs that will be generated to train on. See DataFrame.py.
    productVectorSize = 2 # The dimension of the Product Vectors
    productValueLow = 0.0 # The lowest possible value that a value in a Product Vector can take on
    productValueHigh = 1 # The highest possible value that a value in a Product Vector can take on

    # Mapping Operator Paremeters
    backingTensorDepth = 8 # The depth of the backing tensor that the Mapping Operator represents. Think of this as layers of weights
    backingTensorValueLow = -1.0 # The lowest possible weight value in a backing tensor
    backingTensorValueHigh = 1.0 # The highest possible weight in a backing tensor

    # Performance Metrics
    totalGenerations = -1.0 # A count of how many generations were run
    bestFitness = 99999999 # The best fitness at the end of a GA run
    averageFitness = 99999999 # The average fitness over a GA run
    averageNumberOfGenerationsBetweenFitnessGains = 99999999 # In the name
    generationCountsBetweenFitnessGains = [] # A list of generation counts between fitness gains
    averageFitnessGain = 99999999 # The average fitness gain per generation over the life of the GA
    fitnessGains = [] # A list of fitness gains the algorithm made

    def __init__(self) :
        # Generate and set a default data frame
        self.generateAndSetNewDataFrame()

    def generateAndSetNewDataFrame(self) :
        self.dataFrame = DataFrame(self)

    def getDataFrame(self) :
        return self.dataFrame

    def getBestMappingOperator(self) :
        return self.bestMappingOperator

    def setBestMappingOperator(self, best) :
        self.bestMappingOperator = best
        
    # Performance Metrics Functions
    # ----------------------------------------------------------------------------------------------------------------
    def getTotalGenerations(self) :
        return self.totalGenerations

    def setTotalGenerations(self, totalGenerations) :
        self.totalGenerations = totalGenerations

    def getBestFitness(self) :
        return self.bestFitness

    def setBestFitness(self, bestFitness) :
        self.bestFitness = bestFitness

    def getAverageFitness(self) :
        return self.averageFitness

    def setAverageFitness(self, averageFitness) :
        self.averageFitness = averageFitness

    def addGenerationCountAtFitnessGain(self, count) :
        self.generationCountsBetweenFitnessGains.append(count)

        average = 0
        for generationCount in self.generationCountsBetweenFitnessGains :
            average = average + generationCount
        self.averageNumberOfGenerationsBetweenFitnessGains = average / len(self.generationCountsBetweenFitnessGains)

    def addFitnessGain(self, gain) :
        self.fitnessGains.append(gain)

        average = 0
        for fitnessGain in self.fitnessGains :
            average = average + fitnessGain
        self.averageFitnessGain = average / len(self.fitnessGains)