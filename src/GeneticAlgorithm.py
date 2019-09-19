
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
            if currentBestFitness < bestFitness :
                bestFitness = currentBestFitness
                
            generationCount = generationCount + 1

    def generatePopulation(self) :
        return 0
        # IMPLEMENT ME

    def select(self, population) :
        return 0
        # IMPLEMENT ME

    def crossover(self, population) :
        return 0
        # IMPLEMENT ME

    def mutate(self, population) :
        return 0
        # IMPLEMENT ME

    def evaluatePopulation(self, population) :
        return 0

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

