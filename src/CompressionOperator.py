
import random as random

import tensorflow as tf

class CompressionOperator :

    # Configuration
    # -------------
    backingTensorDepth = 6
    backingTensorValueLow = 0.0
    backingTensorValueHigh = 1.0
    
    # Fitness
    fitness = 99999999

    # Product Vector Dimensions
    #   The Product vector is one dimensional in this implementation
    productVectorSize = None

    # Backing Tensor
    # --------------
    #   The backing tensor is made up of a set of rank 1 tensors (vectors).
    #   Their dimensions are always (in this implementation) 1 X N where
    #   N is the product vector size.0
    #   The backing tensor can also be thought of as a rank 2 tensor where
    #   The Z dimension is described by the successive ordering of the set
    #   of rank 1 tensors.
    backingTensor = []
    backingTensorBiases = []

    def __init__(self, productVectorSize) :
        self.productVectorSize = productVectorSize
        self.generateRandomBackingTensor()

    def generateRandomBackingTensor(self) :

        # Generate the random backing tensor
        self.backingTensor.clear()
        for i in range(self.backingTensorDepth) :
            backingTensorLayer = []
            for j in range(self.productVectorSize) :
                backingTensorLayer.append(random.uniform(self.backingTensorValueLow, self.backingTensorValueHigh))
            self.backingTensor.append(backingTensorLayer)

        # Generate random biases
        self.backingTensorBiases.clear()
        for i in range(self.backingTensorDepth) :
            self.backingTensorBiases.append(random.uniform(self.backingTensorValueLow, self.backingTensorValueHigh))
            
    def mutate(self, mutationRate, mutationMagnitude, topologicalMutationRate, valueReplacementBias) :

        # Randomly add a new layer
        # addALayerChance = random.uniform(0, 1)
        # if addALayerChance < topologicalMutationRate :

        #     # Generate the randomized new layer
        #     newLayer = []
        #     for i in range(self.productVectorSize) :
        #         newLayer.append(random.uniform(self.backingTensorValueLow, self.backingTensorValueHigh))

        #     # Insert the new layer
        #     randomInsertionIndex = random.randint(0, len(self.backingTensor) - 1)
        #     self.backingTensor.insert(randomInsertionIndex, newLayer)

        #     # Generate the randomized new bias value
        #     newBias = random.uniform(self.backingTensorValueLow, self.backingTensorValueHigh)
            
        #     # Insert the new bias at the same index
        #     self.backingTensorBiases.insert(randomInsertionIndex, newBias)

        # # Randomly remove a layer
        # removeALayerChance = random.uniform(0, 1)
        # if removeALayerChance < topologicalMutationRate and len(self.backingTensor) > 2:

        #     randomDeletionIndex = random.randint(0, len(self.backingTensor) - 1)

        #     # Delete the backing tensor layer at the deletion index
        #     del self.backingTensor[randomDeletionIndex]

        #     # Delete the bias value at the deletion index
        #     del self.backingTensorBiases[randomDeletionIndex]
            
        # Randomly mutate the values in the backing tensor
        for i in range(len(self.backingTensor)) :
            rank1Tensor = self.backingTensor[i]
            for j in range(len(rank1Tensor)) :
                if random.uniform(0, 1) < mutationRate :
                    if random.uniform(0, 1) < valueReplacementBias :
                        rank1Tensor[j] = random.uniform(self.backingTensorValueLow, self.backingTensorValueHigh)
                    else :
                        if random.uniform(0, 1) < .5 :
                            rank1Tensor[j] = rank1Tensor[j] + self.mutationMagnitude
                        else :
                            rank1Tensor[j] = rank1Tensor[j] - self.mutationMagnitude

        # Randomly mutate the bias values
        for i in range(len(self.backingTensorBiases)) :
            if random.uniform(0, 1) < mutationRate :
                if random.uniform(0, 1) < valueReplacementBias :
                    self.backingTensorBiases[i] = random.uniform(self.backingTensorValueLow, self.backingTensorValueHigh)
                else :
                    adjustedBias = self.backingTensorBiases[i]
                    
                    if random.uniform(0, 1) < .5 :
                        adjustedBias = adjustedBias + self.mutationMagnitude
                    else :
                        adjustedBias = adjustedBias - self.mutationMagnitude

                    if adjustedBias > self.backingTensorValueLow and adjustedBias < self.backingTensorValueHigh :
                        self.backingTensorBiases[i] = adjustedBias
                        
    def setFitness(self, fitness) :
        self.fitness = fitness

    def getFitness(self) :
        return self.fitness

    def setProductVectorSize(self, productVectorSize) :
        self.productVectorSize = productVectorSize

    def getProductVectorSize(self) :
        return self.productVectorSize

    def getBackingTensor(self) :
        return self.backingTensor

    def getBackingTensorBiases(self) :
        return self.backingTensorBiases

    def setBackingTensor(self, newBackingTensor) :
        self.backingTensor = newBackingTensor

    def setBackingTensorBiases(self, newBiases) :
        self.backingTensorBiases = newBiases

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
