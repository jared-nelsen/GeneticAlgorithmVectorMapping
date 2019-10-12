
from DataFrame import DataFrame

class EvaluationModule :

    # Data Frame
    # ----------
    dataFrame = None

    # Algorithm Parameters
    # --------------------
    
    # GA Parameters
    populationSizeFactor = 5
    maxGenerations = 1000000000
    crossoverRate = .9
    mutationRate = .2
    topologicalMutationRate = .05
    valueReplacementBias = .2
    elitismWeight = .3

    # Data Frame Parameters
    stimulusProductPairCount = 1
    productVectorSize = 3
    productValueLow = 0.0
    productValueHigh = 1.0

    # Compression Operator Paremeters
    backingTensorDepth = 6
    backingTensorValueLow = 0.0
    backingTensorValueHigh = 1.0
    mutationMagnitude = .000001

    # Performance Metrics
    totalGenerations = 0
    bestFitness = 99999999
    averageFitness = 99999999
    
    averageNumberOfGenerationsBetweenFitnessGains = 99999999
    generationCountsBetweenFitnessGains = []
    averageFitnessGain = 99999999
    fitnessGains = []

    def __init__(self) :
        self.generateAndSetNewDataFrame()

    def generateAndSetNewDataFrame(self) :
        dataFrame = DataFrame(productVectorSize, stimulusProductPairCount)

    def getDataFrame(self) :
        return dataFrame

    def getTotalGenerations(self) :
        return totalGenerations

    def setTotalGenerations(self, totalGenerations) :
        self.totalGenerations = totalGenerations

    def getBestFitness(self) :
        return bestFitness

    def setBestFitness(self, bestFitness) :
        self.bestFitness = bestFitness

    def getAverageFitness(self) :
        return averageFitness

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
