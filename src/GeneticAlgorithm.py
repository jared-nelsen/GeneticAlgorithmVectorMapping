
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

    # Hyperparameters
    crossoverRate = .9
    mutationRate = .01

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

    # Function:
    # --------- 
    #   generatePopulation()
    # --------------------------------------------------------------------------
    # Description:
    # ------------
    #   Generates the population of Compression Operators for the Genetic Algorithm.
    # --------------------------------------------------------------------------
    # Parameters:
    # -----------
    #   None
    # --------------------------------------------------------------------------
    # Returns:
    # --------
    #   Nothing
    # --------------------------------------------------------------------------
    # Explanation:
    # ------------
    #   Delegates the generation of randomly initialized Compression Operators to
    #   the Compression Operator constructor. Each Compression Operator needs to 
    #   know what size the vectors in the Product vector are in order to size 
    #   itself correctly. The Data Frame that is global to this class is the single
    #   source of truth for that value so we pass its value into the Compression 
    #   Operators as we construct them.
    # --------------------------------------------------------------------------
    def generatePopulation(self) :
        
        for i in range(self.populationSize) :
            self.population.append(CompressionOperator(self.dataFrame.getProductVectorSize()))

    def select(self, population) :
        return 0
        # IMPLEMENT ME

    def crossover(self, population) :
        return 0
        # IMPLEMENT ME

    # Function:
    # --------- 
    #   mutate()
    # --------------------------------------------------------------------------
    # Description:
    # ------------
    #   Mutates the members of the population in the Genetic Algorithm.
    # --------------------------------------------------------------------------
    # Parameters:
    # -----------
    #   population - The population as it stands after the Crossover process
    # --------------------------------------------------------------------------
    # Returns:
    # --------
    #   The mutated population
    # --------------------------------------------------------------------------
    # Explanation:
    # ------------
    #   Mutates the numerical and shape features of the Compression Operators in
    #   the population.
    #
    #   This implementation of the mutation algorithm can:
    #   1. Add a layer in the Z dimension of the rank 3 Tensor that is the
    #      Compression Operator.
    #   2. Remove a layer in the Z dimension of the rank 3 Tensor that is the
    #      Compression Operator.
    #   3. Mutate the values in the Compression Operator Tensor
    #
    #   Numerical Mutation methodology:
    #   --------------------------------
    #   The Mutation of the values in the Tensor can be performed under several
    #   different strategies. However, in order to make things simple and to
    #   avoid excessive hyperparameter tuning I will use a balanced strategy.
    #
    #       A value can be:
    #       1. Completely thrown out and replaced with a completely random value
    #       2. Slightly altered with tiny numerical adjustments
    #       
    #       The fact of the matter is that I don't quite know which is objectively
    #       better. There will be an element of experimentation. However, the real
    #       power of the Genetic Algorithm in this incarnation is that it has the
    #       ability to implicity select the best strategy. I am choosing to trust
    #       the algorithm and allow it to use both strategies at its discretion.
    #       
    #       One line of thought to pursue is that the choice of mutation strategy
    #       is likely affected by the evaluation function strategy. Obviously we
    #       are targeting a zero error mapping between the Stimuli and Product
    #       vectors. However, because the values in the Product vectors are 
    #       integers and not floating point values there are concerns that while
    #       the algorithm will converge smoothly, it may never converge exactly.
    #       Numerically speaking, it is beneficial to have a smooth and
    #       continuous optimization curve instead of discretely disjoint values
    #       where things like primes might cause problems in convergence. One
    #       strategy to get the best of both worlds is to apply a flooring
    #       function to the output of the Compression Operator Tensor before it
    #       is evaluated. Then integers are evaluated against integers while
    #       allowing the Compression Operator Tensor to be continuously valued.
    # 
    #   More information about Mutation in general:
    #   -------------------------------------------
    #   Mutation occurs based opon the specified mutation rate. It serves to 
    #   exploit the solution members of the population by exploring the search
    #   space that is very local to the member being mutated. In this 
    #   implementation, mutation occurs at the same rate globally to the 
    #   popluation as it does locally to the member. If the member is selected 
    #   to be mutated then the mutation the member undergoes is still dictated
    #   by the same rate. Mutation rates are generally always smaller than crossover
    #   rates and usually very, very small. Some researched-backed values are .01 
    #   to .1 .
    # --------------------------------------------------------------------------
    def mutate(self, population) :
        
        for compressionOperator in population :
            compressionOperator.mutate()

    # Function:
    # --------- 
    #   evaluatePopulation()
    # --------------------------------------------------------------------------
    # Description:
    # ------------
    #   Uses the Data Frame to run and measure the fitness of each Compression
    #   Operator in the population.
    # --------------------------------------------------------------------------
    # Parameters:
    # -----------
    #   population - The population as it stands after the Mutation process
    # --------------------------------------------------------------------------
    # Returns:
    # --------
    #   Nothing
    # --------------------------------------------------------------------------
    # Explanation:
    # ------------
    #   We relegate the logic for running and evaluating a given Compression
    #   Operator to the Data Frame so just pass each Compression Operator into
    #   the Data Frame's evalutation function. The evaluateCompressionOperator()
    #   function in the Data Frame sets the measured fitness value on the 
    #   Compression Operator that is passed into it.
    # --------------------------------------------------------------------------
    def evaluatePopulation(self, population) :
        
        for compressionOperator in population :
            self.dataFrame.evaluateCompressionOperator()

    # Function:
    # --------- 
    #   sortPopulation()
    # ---------------------------------------------------------------------------
    # Description:
    # ------------
    #   Sorts the Compression Operators in the population by their fitness level
    #   in DESCENDING order.
    # ---------------------------------------------------------------------------
    # Parameters:
    # -----------
    #   population - The population as it stands after the Evaluation process
    # ---------------------------------------------------------------------------
    # Returns: 
    # --------
    #   The population sorted by fitness in DESCENDING order.
    # ---------------------------------------------------------------------------
    # Explanation:
    # ------------
    #   Uses Merge Sort to sort the evaluated population to get set up for
    #   the Selection process in the next iteration of the algorithm.
    # ---------------------------------------------------------------------------
    def sortPopulation(self, population) :
        return mergeSort(population)

    # Function:
    # --------- 
    #   mergeSort()
    # ---------------------------------------------------------------------------
    # Description:
    # ------------
    #   The sorting portion of the Merge Sort algorithm.
    # ---------------------------------------------------------------------------
    # Parameters:
    # -----------
    #   population - The population as it stands after the Evaluation process.
    # ---------------------------------------------------------------------------
    # Returns: 
    # --------
    #   Portions of the population based on the recursive depth. The shallowest
    #   depth returns the fully sorted population.
    # ---------------------------------------------------------------------------
    # Explanation:
    # ------------
    #   Performs the sorting portion of Merge Sort. Merge Sort is used as the
    #   sorting algorithm to order the Compression Operators in the population
    #   by fitness value.
    #
    #   Merging is done in DESCENDING ORDER
    # ---------------------------------------------------------------------------
    def mergeSort(self, population) :

        if len(population) == 1 :
            return population

        mid = len(population) // 2
        a = population[:mid]
        b = population[mid:]

        sortedA = self.mergeSort(a)
        sortedB = self.mergeSort(b)

        return self.merge(sortedA, sortedB)

    # Function:
    # --------- 
    #   merge()
    # ---------------------------------------------------------------------------
    # Description:
    # ------------
    #   The merging portion of the Merge Sort algorithm.
    # ---------------------------------------------------------------------------
    # Parameters:
    # -----------
    #   a - The first half of the population to be merged
    #   b - The second half of the population to be merged
    # ---------------------------------------------------------------------------
    # Returns: 
    # --------
    #   The merged population halves in sorted order.
    # ---------------------------------------------------------------------------
    # Explanation:
    # ------------
    #   Performs the merging portion of Merge Sort. Merge Sort is used as the
    #   sorting algorithm to order the Compression Operators in the population
    #   by fitness value. This is one of two functions used in the algorithm.
    #
    #   Merges in DESCENDING ORDER!
    # ---------------------------------------------------------------------------
    def merge(self, a, b) :

        merged = []

        while len(a) > 0 and len(b) > 0 :

            if a[0].getFitness() > b[0].getFitness() :
                merged.append(a.pop(0))
            else :
                merged.append(b.pop(0))

        while len(a) > 0 :
            merged.append(a.pop(0))

        while len(b) > 0 :
            merged.append(b.pop(0))

        return merged

