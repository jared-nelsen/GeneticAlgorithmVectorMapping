
import random as random

import CompressionOperator

class DataFrame :

    # Configuration:
    # --------------
    stimulusProductPairCount = 0
    productValueLow = 0
    productValueHigh = 256

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
    #   within the product vector vector. N is also set by the user
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

    def generateRandomDataFrame(self) :

        # Generate the stimuli
        stimulusValue = 1
        for i in range(self.stimulusProductPairCount) :

            stimulus = []
            stimulus.append(stimulusValue)

            self.stimulusVector.append(stimulus)

            stimulusValue = stimulusValue + 1

        # Generate the product vectors
        for i in range(self.stimulusProductPairCount) :
            productVector = []
            for j in range(self.productVectorSize) :
                productVector.append(random.randint(self.productValueLow, self.productValueHigh))

    def loadDataFrameFromFile(self, filePath) :
        return 0
        #Implement Me

    def evaluateCompressionOperator(self, compressionOperator) :
        return 0
        #Implement Me

