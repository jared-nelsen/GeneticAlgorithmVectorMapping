
from GeneticAlgorithm import GeneticAlgorithm
from DataFrame import DataFrame

  # Silence Tensorflow
import tensorflow as tf
tf.disable_v2_behavior()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def runRandomGavcInstance() :

    productVectorSize = 10
    stimulusProductPairCount = 10

    print("Running GAVC...")

    print("\nInstantiating a random Data Frame...")
    
    dataFrame = DataFrame(productVectorSize, stimulusProductPairCount)

    print("\nRandom Data Frame generated!")

    print("\nSetting up the Genetic Algorithm...")
    
    ga = GeneticAlgorithm(dataFrame)

    print("\nGenetic Algorithm set up complete!")

    print("\nRunning this instance of GAVC...\n")

    ga.run()

def runGavc() :
    
    print("\nWelcome to GAVC!\n")

    print("Select an option:")
    print("1) Run an instance of GAVC")
    print("2) Test the gavc implementation")
    option = input()

    if option == "1" :
        runRandomGavcInstance()


if __name__ == '__main__':
    runGavc()
