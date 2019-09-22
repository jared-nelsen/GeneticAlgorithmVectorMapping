
import random as random

class CompressionOperator :

    # Configuration
    # -------------
    backingTensorDepth = 10
    backingTensorValueLow = -99999999
    backingTensorValueHigh = 99999999

    # Fitness
    fitness = 99999999

    # Product Vector Dimensions
    #   The Product vector is one dimensional in this implementation
    productVectorSize = None

    # Backing Tensor
    # --------------
    #   The backing tensor is made up of a set of rank 1 tensors (vectors).
    #   Their dimensions are always (in this implementation) 1 X N where
    #   N is the product vector size.
    #   The backing tensor can also be thought of as a rank 2 tensor where
    #   The Z dimension is described by the successive ordering of the set
    #   of rank 1 tensors.
    backingTensorDepth = 0
    backingTensor = []

    def __init__(self, productVectorSize) :
        self.productVectorSize = productVectorSize
        self.generateRandomBackingTensor()

    def generateRandomBackingTensor(self) :

        for i in range(self.backingTensorDepth) :
            backingTensorLayer = []
            for i in range(self.productVectorSize) :
                backingTensorLayer.append(random.uniform(self.backingTensorValueLow, self.backingTensorValueHigh))

    def mutate(self, mutationRate, mutationMagnitude) :
        
        for i in range(len(self.backingTensor)) :
            rank1Tensor = self.backingTensor[i]
            for j in range(len(rank1Tensor)) :
                if random.uniform(0, 1) < mutationRate :
                    if random.uniform(0, 1) < .5 :
                        rank1Tensor[j] = random.uniform(self.backingTensorValueLow, self.backingTensorValueHigh)
                    else :
                        if random.uniform(0, 1) < .5 :
                            rank1Tensor[j] = rank1Tensor[j] + mutationMagnitude
                        else :
                            rank1Tensor[j] = rank1Tensor[j] + mutationMagnitude

    # Function:
    # --------- 
    #   feedForward()
    # --------------------------------------------------------------------------
    # Description:
    # ------------
    #   Feeds a stimulus value through the backing tensor using matrix
    #   multiplication.
    # --------------------------------------------------------------------------
    # Parameters:
    # -----------
    #   stimulus - The stimulus scalar that will be fed through the backing
    #              tensor
    # --------------------------------------------------------------------------
    # Result:
    # --------
    #   The given stimulus value has been fed through the backing tensor and the
    #   result is returned in the form of a Tensorflow tensor.
    # --------------------------------------------------------------------------
    # Explanation:
    # ------------
    #   This function uses the Tensoflow API to build a compute Graph from the
    #   values in the backing tensor and then runs that Graph. 
    # --------------------------------------------------------------------------
    def feedForward(self, stimulus) :
        return 0
        #Implement Me

    def setFitness(self, fitness) :
        self.fitness = fitness

    def getFitness(self) :
        return self.fitness

    def setProductVectorSize(self, productVectorSize) :
        self.productVectorSize = productVectorSize

    def getBackingTensor(self) :
        return self.backingTensor

    def setBackingTensor(self, newBackingTensor) :
        self.backingTensor = newBackingTensor

    def clone(self) :
        clone = CompressionOperator(self.productVectorSize)

        clone.setFitness(self.fitness)

        clone.setProductVectorSize(self.productVectorSize)

        cloneBackingTensor = []

        for i in range(len(self.backingTensor)) :
            rank1Tensor = []

            backingTensorRank1Tensor = self.backingTensor[i]
            for j in range(len(backingTensorRank1Tensor)) :
                rank1Tensor.append(backingTensorRank1Tensor[j])

            cloneBackingTensor.append(rank1Tensor)

        clone.setBackingTensor(cloneBackingTensor)

        return clone