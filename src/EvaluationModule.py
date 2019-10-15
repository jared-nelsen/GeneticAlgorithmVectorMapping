
from DataFrame import DataFrame

class EvaluationModule :

    # Data Frame
    # ----------
    dataFrame = None

    # Best Compression Operator
    #   ** This is the best Compression Operator that is the result of training
    bestCompressionOperator = None

    # Algorithm Parameters
    #   ** These parameters are the source of truth for the parameters throughout the algorithm
    # --------------------
    
    # GA Parameters
    populationSizeFactor = 5
    maxGenerations = 1000000000
    crossoverRate = .9
    mutationRate = .1
    topologicalMutationRate = .05
    valueReplacementBias = .05
    mutationMagnitudeLow = .00000000001
    mutationMagnitudeHigh = .1
    elitismWeight = .3

    # Data Frame Parameters
    stimulusProductPairCount = 3
    productVectorSize = 1
    productValueLow = 0.0
    productValueHigh = 1.0

    # Compression Operator Paremeters
    backingTensorDepth = 7
    backingTensorValueLow = 0
    backingTensorValueHigh = 1.0

    # Performance Metrics
    totalGenerations = 0
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

    def getBestCompressionOperator(self) :
        return self.bestCompressionOperator

    def setBestCompressionOperator(self, best) :
        self.bestCompressionOperator = best
        
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
