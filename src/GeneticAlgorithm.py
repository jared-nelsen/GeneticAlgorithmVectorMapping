
class GeneticAlgorithm :

    # Population
    populationSize = 100
    population = []
    
    # Fitnesses
    bestFitness = 99999999

    # Generations
    maxGenerations = 99999999

    def run() :

        generatePopulation()
        
        # Main GA algortihm loop
        generationCount = 0
        while generationCount < maxGenerations :

            # Run GA functions 
            selectedPopulation = select(population)
            crossedOverPopulation = crossover(selectedPopulation)
            mutatedPopulation = mutate(crossedOverPopulation)
            evaluatedPopulation = evaluatePopulation(mutatedPopulation)
            population = sortPopulation(evaluatedPopulation)

            # Check fitnesses
            currentBestFitness = FIXME
            if currentBestFiness < bestFitness :
                bestFitness = currentBestFitness
                
            generationCount = generationCount + 1

    def generatePopulation() :
        return 0
        # IMPLEMENT ME

    def select(population) :
        return 0
        # IMPLEMENT ME

    def crossover(population) :
        return 0
        # IMPLEMENT ME

    def mutate(population) :
        return 0
        # IMPLEMENT ME

    def sortPopulation(population) :
        return 9
        # IMPLEMENT ME

    def mergeSort(population) :

        if len(population) == 1 :
            return population

        mid = len(population) // 2
        a = population[:mid]
        b = population[mid:]

        sortedA = mergeSort(a)
        sortedB = mergeSort(b)

        return merge(sortedA, sortedB)

    def merge(a, b) :

        merged = []

        while len(a) > 0 and len(b) > 0 :

            if a[0] > b[0] :
                merged.append(a.pop(0))
            else :
                merged.append(b.pop(0))

        while len(a) > 0 :
            merged.append(a.pop(0))

        while len(b) > 0 :
            merged.add(b.pop(0))

        return merged

x = [3, 9, 1]

mergeSort(x)

print(y)
