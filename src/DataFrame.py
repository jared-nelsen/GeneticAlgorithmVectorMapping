
import random as random

import tensorflow as tf

import CompressionOperator

class DataFrame :

    # Configuration:
    # --------------
    stimulusProductPairCount = 0
    productValueLow = 0
    productValueHigh = 257

    # Stimuli:
    # --------
    #   The stimuli are rank 0 tensors or can be considered scalars. The actual
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

    # Stimulus - Product pair example:
    # ---------------------------------
    #
    #   stimulusProductPairCount = 3
    #   productVectorSize = 4
    #
    #       Stimulus Vector     Product Vector
    #       [                   [
    #        [1], -------------> [83, 96, 81, 37],
    #        [2], -------------> [76, 1,  33, 44].
    #        [3], -------------> [9,  34. 11, 244]
    #           ]               ]

    def __init__(self, productVectorSize, stimulusProductPairCount) :
        self.productVectorSize = productVectorSize
        self.stimulusProductPairCount = stimulusProductPairCount
        self.generateRandomDataFrame()

    def evaluateCompressionOperator(self, compressionOperator) :

        tf.compat.v1.enable_eager_execution()
        
        # Designate a list of errors that are the result of a stimulus
        # being applied to the mapping operation and measured against
        # the corresponding product vectors
        stimulusProductPairErrors = []

        backingTensor = compressionOperator.backingTensor

        # For each stimulus-product vector pair
        for pairIndex in range(len(self.stimulusVector)) :

            stimulus = self.stimulusVector[pairIndex]
            productVector = self.productVectors[pairIndex]

            # Run the mapping operator with the given stimulus
            # resultantMappingOperationProduct = sess.run(mappingOperation, feed_dict = {inputPlaceholder: stimulus})
            resultantMappingOperationProduct = tf.multiply(stimulus, backingTensor[0])
            for index in range(1, len(backingTensor)) :
                resultantMappingOperationProduct = tf.multiply(resultantMappingOperationProduct, backingTensor[index])
            
            # Scale the values by the highest possible value of the product vector
            resultantMappingOperationProduct = tf.multiply(resultantMappingOperationProduct, self.productValueHigh)

            # Floor the values so as to compare only integers
            resultantMappingOperationProduct = tf.math.floor(resultantMappingOperationProduct)

            # Compare the error between the resultant product and the given product
            stimulusProductPairError = tf.compat.v1.losses.absolute_difference(resultantMappingOperationProduct, productVector)

            # Record the error
            stimulusProductPairErrors.append(stimulusProductPairError)

        # Compute the sum total of errors in this compression operator evaluation operation
        sumOfErrors = tf.reduce_sum(stimulusProductPairErrors)

        # Set the sum of the errors over this compression operators as the fitness of
        # the compression operator
        sumOfErrors = sumOfErrors.numpy()
        compressionOperator.setFitness(sumOfErrors)

        # Reset the default Tensorflow graph
        # tf.reset_default_graph()

    def generateRandomDataFrame(self) :

        # Generate the stimuli
        stimulusValue = 1
        for i in range(self.stimulusProductPairCount) :

            stimulus = []

            # Multiply by 1.0 to make it a float
            stimulus.append(stimulusValue * 1.0)

            self.stimulusVector.append(stimulus)

            stimulusValue = stimulusValue + 1

        # Generate the product vectors
        for i in range(self.stimulusProductPairCount) :
            productVector = []
            for j in range(self.productVectorSize) :
                # Generate a random int but multiply it by 1.0 to make it a float
                productVector.append(random.randint(self.productValueLow, self.productValueHigh) * 1.0)
            self.productVectors.append(productVector)

    def loadDataFrameFromFile(self, filePath) :
        return 0
        #Implement Me

    def writeDataToFile(self, filePath) :
        return 0
        #Implement Me

    def getProductVectorSize(self) :
        return self.productVectorSize

