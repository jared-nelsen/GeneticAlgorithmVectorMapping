
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

# -------------------------------------------------------------
# File:
# -----
#   gavm.py
# -------------------------------------------------------------
# Description:
# ------------
#   The gavm.py file is the main driver for the program. It is
#   responsible for setting up the parameters for the simulation
#   by setting up the algorithm parameters and data parameters in an
#   Evaluation module and passing it to an instance of a Genetic
#   Algorithm. Neural Networks are represented as members of the
#   population of the Genetic Algortihm and are named Mapping
#   Operators. The purpose of a Mapping Operator is to map
#   the set of Stimulus-Product pairs that are generated in the
#   DataFrame object together. See MappingOperator.py for more
#   details.
#
#   The hierarchy of ownership in the simulation is:
#
#       1. gavm.py owns a Genetic Algorithm instance
#       2. GeneticAlgorithm.py owns:
#           - The instance of EvaluationModule.py
#               - Which owns the instance of DataFrame.py
#           - N instances of the MappingOperator.py class where N
#               is the size of the population in the Genetic
#               Algorithm
# -------------------------------------------------------------

def runRandomGavcInstance() :

    evaluationModule = EvaluationModule()
    
    ga = GeneticAlgorithm(evaluationModule)

    print("\nRunning this instance of GAVM...\n")

    ga.run()

def runGavc() :
    
    print("\nWelcome to GAVM!\n")

    print("Select an option:")
    print("1) Run an instance of GAVM with the default parameters found in EvaluationModule.py")
    option = input()

    if option == "1" :
        runRandomGavcInstance()
    else :
        print("Please select a valid option... Exiting...")
        exit()

if __name__ == '__main__':
    runGavc()
