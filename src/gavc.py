
from GeneticAlgorithm import GeneticAlgorithm
from DataFrame import DataFrame
from EvaluationModule import EvaluationModule

  # Silence Tensorflow
import tensorflow as tf
# tf.disable_v2_behavior()
tf.enable_eager_execution()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def runRandomGavcInstance() :

    evaluationModule = EvaluationModule()
    
    ga = GeneticAlgorithm(evaluationModule)

    print("\nRunning this instance of GAVC...\n")

    ga.run()

def runExperimentWithGAVC() :

    evaluationModule = EvaluationModule()
    evaluationModule.maxGenerations = 10

    ga = GeneticAlgorithm(evaluationModule)

    print("Running the default experiment with GAVC...\n")

    ga.run()

    print("\nTraining Finished...")

    inconsistencies = evaluationModule.dataFrame.evaluateFinalCompressionOperator(evaluationModule.bestCompressionOperator)

    print("Number of values off = ", inconsistencies)

def runGavc() :
    
    print("\nWelcome to GAVC!\n")

    print("Select an option:")
    print("1) Run a default instance of GAVC")
    print("2) Run an experiment with GAVC")
    option = input()

    if option == "1" :
        runRandomGavcInstance()
    elif option == "2" :
        runExperimentWithGAVC()

if __name__ == '__main__':
    runGavc()
