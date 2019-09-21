
import random as random

class CompressionOperator :

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
    backingTensor = []

    def __init__(self, productVectorSize) :
        self.productVectorSize = productVectorSize

    def mutate(self, mutationRate, mutationMagnitude, tensorValueLow, tensorValueHigh) :
        
        for i in range(len(self.backingTensor)) :
            rank1Tensor = self.backingTensor[i]
            for j in range(len(rank1Tensor)) :
                if random.uniform(0, 1) < mutationRate :
                    if random.uniform(0, 1) < .5 :
                        rank1Tensor[j] = random.uniform(tensorValueLow, tensorValueHigh)
                    else :
                        if random.uniform(0, 1) < .5 :
                            rank1Tensor[j] = rank1Tensor[j] + mutationMagnitude
                        else :
                            rank1Tensor[j] = rank1Tensor[j] + mutationMagnitude

    def setFitness(self, fitness) :
        self.fitness = fitness

    def getFitness(self) :
        return self.fitness

    def getBackingTensor(self) :
        return self.backingTensor

    def setBackingTensor(self, newBackingTensor) :
        self.backingTensor = newBackingTensor

    def clone(self) :
        clone = CompressionOperator(self.productVectorSize)

        cloneBackingTensor = []

        for i in range(len(self.backingTensor)) :
            rank1Tensor = []

            backingTensorRank1Tensor = self.backingTensor[i]
            for j in range(len(backingTensorRank1Tensor)) :
                rank1Tensor.append(backingTensorRank1Tensor[j])

            cloneBackingTensor.append(rank1Tensor)

        clone.setBackingTensor(cloneBackingTensor)

        return clone