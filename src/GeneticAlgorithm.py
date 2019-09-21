
import random as random

from DataFrame import DataFrame
from CompressionOperator import CompressionOperator

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
    mutationMagnitude = .001
    tensorValueLow = -99999999
    tensorValueHigh = 99999999

    def __init__(self, dataFrame) :

        #Set the data frame for this GA instance
        self.dataFrame = dataFrame

    def run(self) :

        # Generate the population
        self.generatePopulation()
        # Evaluate it for the first time
        self.evaluatePopulation()
        
        # Main GA algortihm loop
        generationCount = 0
        while generationCount < self.maxGenerations :

            # Run GA functions 
            self.select()
            self.crossover()
            self.mutate()
            self.evaluatePopulation()
            self.sortPopulation()

            # Check fitnesses
            currentBestFitness = self.population[-1].getFitness()
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
    # Result:
    # --------
    #   The global population has been generated
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

    # Function:
    # --------- 
    #   select()
    # --------------------------------------------------------------------------
    # Description:
    # ------------
    #   Selects members of the population to cross over.
    #
    #   Relies in the population being sorted by fitness value in DESCENDING order
    #   at this point
    # --------------------------------------------------------------------------
    # Parameters:
    # -----------
    #   None
    # --------------------------------------------------------------------------
    # Result:
    # --------
    #   A new population of selected members has been generated. The new population 
    #   is double the size of the original population. This new population will be
    #   made up of consecutive parents that are to be crossed over.
    #
    #   Ordering Schema:
    #   ----------------
    #   [Parent A1][Parent B1] , [Parent A2][Parent B2] , ... , [Parent AN][Parent BN]
    #
    #   Each pair of parents A and B for a given selection will be crossed over
    #   to form a new population that is of the original length indicated by the
    #   global variable populationSize. All of this is accomplished in the crossover
    #   function.
    #
    # --------------------------------------------------------------------------
    # Explanation:
    # ------------
    #   This implementation of the selection algorithm uses Roulette Wheel Selection.
    #   The goal of the algorithm is to select members in a manner that is
    #   proportionate to their fitness. In short, the fitter members get a larger
    #   piece of the probability pie; they have a higher likelihood of being
    #   selected.
    #
    #   This algorithm only works in the population at the beginning of it is
    #   sorted in DESCENDING order
    #
    #   More information about selection in general:
    #   --------------------------------------------
    #   Roulette Wheel selection is a preferred method in Genetic Algorithms because
    #   though the members with the best fitness have a greater chance of being 
    #   selected for crossover, the poorer members still have some chance which is
    #   important for retaining as much information in the population as possible as
    #   well as to effectively explore the search space.
    #
    #   Useful links:
    #   -------------
    #   A good explanation of various selection strategies:
    #       https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
    # --------------------------------------------------------------------------
    def select(self) :

        selectedPopulation = []

        fitnessSumOfPopulation = 0
        for compressionOperator in self.population :
            fitnessSumOfPopulation = fitnessSumOfPopulation + compressionOperator.getFitness()

        for i in range(len(self.population) * 2) : # Notice the doubling of the population
            randomFitness = random.uniform(0, fitnessSumOfPopulation)
            localFitnessSum = 0
            for j in range(len(self.population)) :
                localFitnessSum = localFitnessSum + self.population[j].getFitness()
                if localFitnessSum > randomFitness :
                    selectedPopulation.append(self.population[j])

        self.population = selectedPopulation

    # Function:
    # --------- 
    #   crossover()
    # --------------------------------------------------------------------------
    # Description:
    # ------------
    #   Crosses over the Compression Operators in the fitness-selected population
    # --------------------------------------------------------------------------
    # Parameters:
    # -----------
    #   None
    # --------------------------------------------------------------------------
    # Result:
    # --------
    #   A new resized population that has been crossed over from the selected
    #   parents in the population at the beginning of this function.
    #
    #   The result is that the genetic information from each parent 2-tuple
    #   has been assimilated into one single new child. The set of these
    #   children becomes the new population.
    # --------------------------------------------------------------------------
    # Explanation:
    # ------------
    #   This algorithm takes the ordered tuples of parents in the fitness-selected
    #   population  and crosses over their numerical features.
    #
    #   Crossover methodology:
    #   ----------------------
    #   This first iteration of the crossover algorithm only crosses over the
    #   values in the Compression Operator Tensor by simply picking one of
    #   the two values in each parent to pass on to the single child. This
    #   random picking happens at a rate represented by the crossover rate
    #   that is global to the Genetic Algorithm.
    #
    #       Methodology example:
    #       --------------------
    #       Parent A Tensor: [1, 2, 3, 9]
    #       Parent B Tensor: [4, 5, 6, 8]
    #       Child Tensor:    [1, 5, 3, 8]
    #
    #   General Crossover information:
    #   ------------------------------
    #   Crossover occurs based on the specified crossover rate. Crossover rates are 
    #   meant to be high and usually always higher than mutation rates. Some 
    #   research-backed values are .8 - .99. Crossover is representative of the
    #   exploratory element of the algorithm. It seeks to deviate enough from the 
    #   current best solutions in order to find 'distant' solutions that may be 
    #   better. It does this by recombinating the encodings of the population of 
    #   solutions with each other. Solutions with a higher fitness have a 
    #   commensurately better chance of passing on their information. This logic is
    #   carried out in the select routine. However, there is still a chance that a 
    #   better solution lies 'closer' to a current poor one. The GA takes this into 
    #   account by recombinating solutions with each other. This methodology allows
    #   these properties: a graceful approch to a better solution, consideration
    #   that a poorer solution may contain effective components, consideration that
    #   it is logical that better solutions working together will probably result in
    #   an even better solution, extreme explorations should be mitigated, and 
    #   exploration should be made on logical (good fitness) bases. Note that the 
    #   population size passed to this routine should be exactly double that of the
    #   working population because the generation of N childred requires N * 2 parents.     
    #
    # --------------------------------------------------------------------------
    def crossover(self) :

        crossedOverPopulation = []
        
        for i in range(len(self.population)) :

            parentA = self.population[i]
            i = i + 1
            parentB = self.population[i]

            newPopulationMemberBackingTensor = []
            parentABackingTensor = parentA.getBackingTensor()
            parentBBackingTensor = parentB.getBackingTensor()

            for j in range(len(newPopulationMemberBackingTensor)) :

                newPopulationMemberBackingTensorRank1TensorMember = []
                parentABackingTensorRank1TensorMember = parentABackingTensor[j]
                parentBBackingTensorRank1TensorMember = parentBBackingTensor[j]

                for k in range(len(parentABackingTensorRank1TensorMember)) : 

                    # Notice we default to parent A's data
                    if random.uniform(0, 1) < self.crossoverRate :
                        newPopulationMemberBackingTensorRank1TensorMember.append(parentBBackingTensorRank1TensorMember[k])
                    else :
                        newPopulationMemberBackingTensorRank1TensorMember.append(parentABackingTensorRank1TensorMember[k])

                newPopulationMemberBackingTensor.append(newPopulationMemberBackingTensorRank1TensorMember)    

            newPopulationMember = CompressionOperator(parentA.getProductVectorSize())
            newPopulationMember.setBackingTensor(newPopulationMemberBackingTensor)

            crossedOverPopulation.append(newPopulationMember)


        self.population = crossedOverPopulation


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
    #   None
    # --------------------------------------------------------------------------
    # Result:
    # --------
    #   The global population has been mutated
    # --------------------------------------------------------------------------
    # Explanation:
    # ------------
    #   Mutates the numerical and shape features of the Compression Operators in
    #   the population.
    #
    #   The current implementation of the mutation algortithm can:
    #   ----------------------------------------------------------
    #   1. Mutate the values in the Compression Operator Tensor
    #
    #   An ideal future implementation of the mutation algorithm can:
    #   -------------------------------------------------------------
    #   1. Add a layer in the Z dimension of the rank 3 Tensor that is the
    #      Compression Operator.
    #   2. Remove a layer in the Z dimension of the rank 3 Tensor that is the
    #      Compression Operator.
    #   3. Mutate the values in the Compression Operator Tensor
    #
    #       ** This ideal mutation strategy will affect how crossover is
    #          performed.
    #
    #   A more advanced form of mutation can:
    #   -------------------------------------
    #   1. Mutate mutation hyperparameters:
    #       1. tensorValueLow
    #       2. tensorValueHigh
    #       3. mutationMagnitude
    #       4. ratioBetweenValueReplacementAndValueAdjustment
    #
    #       ** This strategy would require a refactor to allow each Compression
    #          Operator value to hold its own mutation hyperparameters
    #
    #   Numerical Mutation methodology:
    #   --------------------------------
    #   The Mutation of the values in the Tensor can be performed under several
    #   different strategies. However, in order to make things simple and to
    #   avoid excessive hyperparameter tuning I will use a balanced strategy.
    #
    #       A value can be:
    #       ---------------
    #       1. Replaced: The value is completely thrown out and replaced with a
    #                    completely within the range of possible tensor values 
    #       2. Adjusted: The value is altered with tiny numerical adjustments
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
    def mutate(self) :
        
        for compressionOperator in self.population :
            compressionOperator.mutate(self.mutationRate,
                                       self.mutationMagnitude,
                                       self.tensorValueLow,
                                       self.tensorValueHigh)

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
    #   None
    # --------------------------------------------------------------------------
    # Result:
    # --------
    #   The global population has had each of their fitnesses evaluated
    # --------------------------------------------------------------------------
    # Explanation:
    # ------------
    #   We relegate the logic for running and evaluating a given Compression
    #   Operator to the Data Frame so just pass each Compression Operator into
    #   the Data Frame's evalutation function. The evaluateCompressionOperator()
    #   function in the Data Frame sets the measured fitness value on the 
    #   Compression Operator that is passed into it.
    # --------------------------------------------------------------------------
    def evaluatePopulation(self) :
        
        for compressionOperator in self.population :
            self.dataFrame.evaluateCompressionOperator(compressionOperator)

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
    #   None
    # ---------------------------------------------------------------------------
    # Result: 
    # --------
    #   The population has been sorted by fitness in DESCENDING order.
    # ---------------------------------------------------------------------------
    # Explanation:
    # ------------
    #   Uses Merge Sort to sort the evaluated population to get set up for
    #   the Selection process in the next iteration of the algorithm.
    # ---------------------------------------------------------------------------
    def sortPopulation(self) :
        self.population = self.mergeSort(self.population)

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
    #   mergingPopulation - The population as it stands after the Evaluation process.
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
    def mergeSort(self, mergingPopulation) :

        if len(mergingPopulation) == 1 :
            return mergingPopulation

        mid = len(mergingPopulation) // 2
        a = mergingPopulation[:mid]
        b = mergingPopulation[mid:]

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

