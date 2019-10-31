
import random as random

import tensorflow as tf

import MappingOperator

# -------------------------------------------------------------
# File:
# -----
#   DataFrame.py
# -------------------------------------------------------------
# Description:
# ------------
#   The DataFrame file contains the DataFrame class. The
#   DataFrame class is responsible for two things:
#   
#       1. Contain the Stimulus-Product pairs
#       2. Evaluate Mapping Operators against the Stimulus
#          Product pairs contained in the Data Frame
# -------------------------------------------------------------

class DataFrame :

    # Configuration:
    # --------------
    stimulusProductPairCount = 0
    productValueLow = 0
    productValueHigh = 1.0

    # Stimuli:
    # --------
    #   The stimuli are rank 0 tensors and can be considered scalars. The actual
    #   values of the stimuli are irrelevant. We take advantage of this fact to
    #   generalize the algorithm so as not to store stimuli vectors but instead
    #   complute them in a particular manner. To keep things simple, we just
    #   set the values of the stimuli vector to be the integers from 1 to
    #   N where N is the number of stimulus-product pairs.
    #
    #   Stimulus vector examples:
    #   -------------------------
    #   
    #   stimulusProductPairCount = 3
    #       [[1], [2], [3]]
    #
    #   stimulusProductPairCount = 5
    #       [[1], [2], [3], [4], [5]]
    #
    stimulusVector = []

    # Product Vectors:
    # ----------------
    #   The product vectors are rank 1 tensors. They are sized according to how
    #   large the data the user wants to compress is. There are N product vectors
    #   within the product vector vector. N is also set by the user.
    #
    #   Product vector example:
    #   -----------------------
    #   stimulusProductPairCount = 3
    #   productVectorSize = 2
    #   
    #       [[6, 4],
    #        [2, 3]
    #        [9, 7]]
    #
    productVectorSize = 0
    productVectors = []

    # Stimulus - Product pair Mapping example:
    # ---------------------------------
    #
    #   stimulusProductPairCount = 3
    #   productVectorSize = 4
    #
    #       Stimulus Scalars                 Product Vectors
    #       [                                [
    #        [1], --- Mapping Operator X ---> [83, 96, 81, 37],
    #        [2], --- Mapping Operator X ---> [76, 1,  33, 44].
    #        [3], --- Mapping Operator X ---> [9,  34. 11, 244]
    #       ]                                ]
    #
    #       Note that the Mapping Operator X is the same! We are
    #       mapping the **set** of Stimuli to the **set** of
    #       Product Vectors with a **single** Mapping Operator!

    def __init__(self, evaluationModule) :

        # Set the Data Frame paramters from the evaluation module
        self.stimulusProductPairCount = evaluationModule.stimulusProductPairCount
        self.productValueLow = evaluationModule.productValueLow
        self.productValueHigh = evaluationModule.productValueHigh
        self.productVectorSize = evaluationModule.productVectorSize
        
        # Generate a random data frame
        self.generateRandomDataFrame()

    # Function:
    # --------- 
    #   evaluateMappingOperator()
    # --------------------------------------------------------------------------
    # Description:
    # ------------
    #   Simulates a neural network from the given Mapping Operator by feeding
    #   it the Stimulus Value and evaluating the result against the corresponding
    #   Product vector.
    # --------------------------------------------------------------------------
    # Parameters:
    # -----------
    #   mappingOperator - The Mapping Operator to be evaluated
    # --------------------------------------------------------------------------
    # Result:
    # --------
    #   The given Mapping Operator has been evaluated against the data contained
    #   within this DataFrame.
    # --------------------------------------------------------------------------
    # Explanation:
    # ------------
    #   This function uses Tensorflow to simulate a neural network where the
    #   backing tensor in the given Mapping Operator are the weights of the
    #   network. Recall that it is the Genetic Algorithm's responsibility to
    #   evolve the weights of the Mapping Operators in its population to get
    #   a better and better solution. This function serves as the measure for
    #   how well the Mapping Operator maps the given Stimulus Values to the
    #   given Product vectors. See the Stimulus-Product Pair example above to
    #   get a graphical respresentation of the idea.
    # --------------------------------------------------------------------------
    def evaluateMappingOperator(self, mappingOperator) :
        
        # Designate a list of errors that are the result of a stimulus
        # being applied to the mapping operation and measured against
        # the corresponding product vectors
        stimulusProductPairErrors = []

        backingTensor = mappingOperator.getBackingTensor()
        backingTensorBiases = mappingOperator.getBackingTensorBiases()
        
        # For each stimulus-product vector pair
        for pairIndex in range(len(self.stimulusVector)) :

            stimulus = self.stimulusVector[pairIndex]
            productVector = self.productVectors[pairIndex]

            # Simulate a neural network
            resultantMappingOperationProduct = tf.nn.leaky_relu(tf.add(tf.scalar_mul(stimulus, backingTensor[0]), backingTensorBiases[0]))
            for i in range(1, len(backingTensor)) :
                resultantMappingOperationProduct = tf.nn.leaky_relu(tf.add(tf.matmul(resultantMappingOperationProduct, backingTensor[i]), backingTensorBiases[i]))

            productVector = productVector.numpy()
            resultantMappingOperationProduct = resultantMappingOperationProduct.numpy()
            stimulusProductPairError = 0
            for i in range(len(productVector)) :
                stimulusProductPairError = stimulusProductPairError + abs(productVector[i] - resultantMappingOperationProduct[i])
            
            # Record the error
            stimulusProductPairErrors.append(stimulusProductPairError)

        # Compute the sum total of errors in this compression operator evaluation operation
        sumOfErrors = tf.reduce_sum(stimulusProductPairErrors)

        # Set the sum of the errors over this compression operators as the fitness of
        # the mapping operator
        sumOfErrors = sumOfErrors.numpy()
        mappingOperator.setFitness(sumOfErrors)

    def evaluateFinalMappingOperator(self, finalMappingOperator) :

        tf.enable_eager_execution()

        numberOfIncorrectValues = 0

        backingTensor = finalMappingOperator.getBackingTensor()
        backingTensorBiases = finalMappingOperator.getBackingTensorBiases()

        for pairIndex in range(len(self.stimulusVector)) :

            stimulus = self.stimulusVector[pairIndex]
            productVector = self.productVectors[pairIndex]

            # Run the neural network
            resultantMappingOperationProduct = tf.nn.leaky_relu(tf.add(tf.scalar_mul(stimulus, backingTensor[0]), backingTensorBiases[0]))
            for i in range(1, len(backingTensor)) :
                resultantMappingOperationProduct = tf.nn.leaky_relu(tf.add(tf.matmul(resultantMappingOperationProduct, backingTensor[i]), backingTensorBiases[i]))

            productVector = productVector.numpy()
            resultantProductValues = resultantMappingOperationProduct.numpy()
            
            # Detect differences between the vector values
            for valueIndex in range(len(productVector)) :
                if productVector[valueIndex] != resultantProductValues[valueIndex] :
                    numberOfIncorrectValues = numberOfIncorrectValues + 1

        return numberOfIncorrectValues
    
    def generateRandomDataFrame(self) :

        # Generate the stimuli
        stimulusValue = 1
        for i in range(self.stimulusProductPairCount) :

            # Multiply by 1.0 to make it a float
            stimulus = stimulusValue * 1.0
            self.stimulusVector.append(stimulus)

            stimulusValue = stimulusValue + 1

        # Generate the product vectors
        for i in range(self.stimulusProductPairCount) :
            self.productVectors.append(tf.random.uniform([self.productVectorSize, 1], minval = self.productValueLow, maxval = self.productValueHigh))

    def loadDataFrameFromFile(self, filePath) :
        return 0
        #Implement Me

    def writeDataToFile(self, filePath) :
        return 0
        #Implement Me

    def getProductVectorSize(self) :
        return self.productVectorSize

