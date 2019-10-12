
from GeneticAlgorithm import GeneticAlgorithm
from DataFrame import DataFrame
from EvaluationModule import EvaluationModule

  # Silence Tensorflow
import tensorflow as tf
# tf.disable_v2_behavior()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def runRandomGavcInstance() :

    evaluationModule = EvaluationModule()
    
    ga = GeneticAlgorithm(evaluationModule)

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
