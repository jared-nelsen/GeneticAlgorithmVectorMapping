
from DataFrame import DataFrame

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
    populationSizeFactor = 5
    maxGenerations = 1000000000
    crossoverRate = .8
    mutationRate = .1
    mutationLikelihood = .1
    biasMutationLikelihood = .1
    topologicalMutationRate = 0
    valueReplacementBias = .05
    mutationMagnitudeLow = .00000001
    mutationMagnitudeHigh = .0001
    elitismWeight = 0

    # GA Algorithm Parameters
    selectionMethodIndicator = 0
    rouletteWheelselectionBias = .08
    tournamentPopulationProportion = .3
    
    # Data Frame Parameters
    stimulusProductPairCount = 2
    productVectorSize = 2
    productValueLow = 0.0
    productValueHigh = 1

    # Mapping Operator Paremeters
    backingTensorDepth = 8
    backingTensorValueLow = -1.0
    backingTensorValueHigh = 1.0

    # Performance Metrics
    totalGenerations = -1.0
    bestFitness = 99999999
    averageFitness = 99999999
    averageNumberOfGenerationsBetweenFitnessGains = 99999999
    generationCountsBetweenFitnessGains = []
    averageFitnessGain = 99999999
    fitnessGains = []

    def __init__(self) :
        # Generate and set a default data frame
        # Call this function again to set the right parameters on the data frame
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
