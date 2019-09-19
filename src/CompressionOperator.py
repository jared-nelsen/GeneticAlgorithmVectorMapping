
class CompressionOperator :

    # Fitness
    fitness = 99999999

    # The Product vector is one dimensional in this implementation
    productVectorSize = None

    def __init__(self, productVectorSize) :
        self.productVectorSize = productVectorSize

    def setFitness(self, fitness) :
        self.fitness = fitness
