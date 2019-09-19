
import DataFrame
import CompressionOperator

class GeneticAlgorithm :

    # Data Frame
    dataFrame = None

    # Population
    population = []
    populationSize = 100
    
    # Fitnesses
    bestFitness = 99999999

    # Generations
    maxGenerations = 1000

    def __init__(self, dataFrame) :

        #Set the data frame for this GA instance
        self.dataFrame = dataFrame

    def run(self) :

        self.generatePopulation()
        
        # Main GA algortihm loop
        generationCount = 0
        while generationCount < self.maxGenerations :

            # Run GA functions 
            selectedPopulation = self.select(self.population)
            crossedOverPopulation = self.crossover(selectedPopulation)
            mutatedPopulation = self.mutate(crossedOverPopulation)
            evaluatedPopulation = self.evaluatePopulation(mutatedPopulation)
            self.population = self.sortPopulation(evaluatedPopulation)

            # Check fitnesses
            currentBestFitness = 0 #FIX ME
            if currentBestFitness < self.bestFitness :
                self.bestFitness = currentBestFitness
                
            generationCount = generationCount + 1

    # Function: generatePopulation()
    # Generates the population of Compression Operators for the Genetic Algorithm.
    # Each Compression Operator needs to know what size the vectors in the Product
    # vector are in order to size itself correctly. The Data Frame that is global
    # to this class is the single source of truth for that value so we pass its
    # value into the Compression Operators as we construct them.
    def generatePopulation(self) :
        
        for i in range(self.populationSize) :
            self.population.append(CompressionOperator(self.dataFrame.getProductVectorSize()))

    def select(self, population) :
        return 0
        # IMPLEMENT ME

    def crossover(self, population) :
        return 0
        # IMPLEMENT ME

    def mutate(self, population) :
        return 0
        # IMPLEMENT ME

    # Function: evaluatePopulation()
    # Uses the Data Frame to run and measure the fitness of each Compression
    # Operator in the population.
    # We relegate the logic for running and evaluating a given Compression
    # Operator to the Data Frame so just pass each Compression Operator into
    # the Data Frame's evalutation function. The evaluateCompressionOperator()
    # function in the Data Frame sets the measured fitness value on the 
    # Compression Operator that is passed into it.
    def evaluatePopulation(self, population) :
        
        for compressionOperator in population :
            self.dataFrame.evaluateCompressionOperator()

    def sortPopulation(self, population) :
        return 9
        # IMPLEMENT ME

    def mergeSort(self, population) :

        if len(population) == 1 :
            return population

        mid = len(population) // 2
        a = population[:mid]
        b = population[mid:]

        sortedA = self.mergeSort(a)
        sortedB = self.mergeSort(b)

        return self.merge(sortedA, sortedB)

    def merge(self, a, b) :

        merged = []

        while len(a) > 0 and len(b) > 0 :

            if a[0] > b[0] :
                merged.append(a.pop(0))
            else :
                merged.append(b.pop(0))

        while len(a) > 0 :
            merged.append(a.pop(0))

        while len(b) > 0 :
            merged.append(b.pop(0))

        return merged

