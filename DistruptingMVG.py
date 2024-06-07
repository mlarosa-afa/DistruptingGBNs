import sys
import getopt

import Example1_Zillow as ex1
import Example2_Loans as ex2
import Example3_LGSSM as ex3
import Pareto as p

help = """
NAME
       DistruptingMVG - Optimally poisoning Multivariate Gaussians

SYNOPSIS
       DistruptingMVG.py [-v|--version] [-h|--help] [-t|--time-elasped] [-c|--concavity] 
           [-p|--paretofront] exampleProblem method args*
           
DESCRIPTION
       DistruptingMVG provides a methods to poison ML models which can
       interpreted as MVG. This code expands on the work "Indiscriminate Disruption of 
       Conditional Inference on Multivariate Gaussians" by Dr. William Caballero,
       Dr. Alex Fisher, Dr. Vahid Tarokh, and Matthew LaRosa.
    
       exampleProblem:
           exampleProblem specifies which example to run. Valid options are 'Zillow', 'Loan' and 'LGSSM'
       method:
           method specifies which type of attack to perform. Valid options: 'white-box', 'grey-box-SAA', and 'grey-box-SGA'
       args*:
           Additional required parameters based on method. 
            1. white-box -- u_1
            2. grey-box-SAA--u_1, numsamples, Psi coefficent, mu_not coefficent, Kappa, nu
            3. grey-box-SGA--u_1, method, Psi coefficent, mu coefficent, Kappa, nu, learning rate, epsilon
                method options: 1.AdaGrad 2.RMSProp 3.Adam
    
OPTIONS
       --version | -v
           Prints the MVG Example version.

       --help | -h
           Prints the usage and description for running the code.
        
       --time-elapsed | -t
           Prints the time elapsed for each code segment
           
       --concavity | -c
           Prints the guaranteed concavity/convexity range for weight U_1. 
           Only an option when method is 'white-box'
        
       --pareto-front=<file> | -p filename
           Generates a .tsv with data to generate a pareto front. 
           Filename specifies output file. Overrides -c and -t commands.
           argument u_1 is not used.
"""

opts, args = getopt.getopt(sys.argv[1:], "vhtcp:", ["version", "help", "time-elapsed", "concavity", "pareto-front="])

time = False
concavity = False
pareto = False

for opt, arg in opts:
    if opt == "-v" or opt == "--version":
        print("0.2.0")
        quit(1)
    if opt == "-h" or opt == "--help":
        print(help)
        quit(1)
    if opt == "-t" or opt == "--time-elapsed":
        time = True
    if opt == "-c" or opt == "--concavity":
        concavity = True
    if opt == "-p" or opt == "--pareto-front":
        pareto = True
        filename = arg
        time = False
        concavity = False

if args[0] in ['Zillow', 'Loan', 'LGSSM']:
    example = args[0]
else:
    raise Exception("exampleNum argument passed with invalid option. Valid options: 'Zillow', 'Loan', and 'LGSSM'")

if args[1] in ["white-box", "grey-box-SAA", "grey-box-SGA", 'baseline', 'random']:
    method = args[1]
else:
    raise Exception(
        "method argument passed with invalid option. Valid options: 'white-box', 'grey-box-SAA', 'grey-box-SGA', 'baseline', and 'random'.")

#Ensures all parameters passed
if example == "LGSSM" and method in ["white-box", "random", "baseline"] and not (len(args) == 4):
    raise Exception(
        "Arguments passed to 'white-box' for 'LGSSM' are incorrect. Must include only: u_1, risk_tolerance")
elif not(example=="LGSSM") and method in ["white-box", "random", "baseline"] and not (len(args) == 3):
    raise Exception(
        "Arguments passed to 'white-box' are incorrect. Must include only: u_1")
elif example == "LGSSM" and method == "grey-box-SAA" and not(len(args) == 5):
    raise Exception(
        "Arguments passed to 'grey-box-SAA' for 'LGSSM' are incorrect. Must include only: u_1, num_samples, risk_tolerance")
elif not(example == "LGSSM") and method == "grey-box-SAA" and not(len(args) == 8):
    raise Exception(
        "Arguments passed to 'grey-box-SAA' are incorrect. Must include only: u_1, numsamples, Psi coefficent, mu_not coefficent, Kappa, nu")
elif method == "grey-box-SGA" and not(example == 'Zillow'):
    raise Exception(
        "'grey-box-SGA' not available for chosen example. Available SGA Models: 'Zillow'")
elif method == "grey-box-SGA" and not(len(args) >= 10):
    raise Exception(
        "Arguments passed to 'grey-box-SGA' are incorrect. Must include all: u_1, method, Psi coefficent, mu coefficent, Kappa, nu, learn rate, epsilon\n\t where method options are: 1.AdaGrad 2.RMSProp 3.Adam")

args = [float(x) for x in args[2:]]
#checks if a pareto front should be generated
if pareto:
    p.generate_pareto(filename, example, method, args)
    quit(1)

if example == 'Zillow':
    if method == 'baseline':
        ex1.baseline(args[0], concavityFlag=concavity, timeFlag=time)
    elif method == 'random':
        ex1.inst_random(args[0], concavityFlag=concavity, timeFlag=time)
    elif method == "white-box":
        ex1.wb(args[0], concavityFlag=concavity, timeFlag=time)
    elif method == "grey-box-SAA":
        ex1.saa(args[0], int(args[1]), args[2], args[3], args[4], args[5], concavityFlag=concavity, timeFlag=time)
    elif method == "grey-box-SGA":
        ex1.sgd(args[0], args[2], args[3], args[4], args[5], int(args[1]), args[6], args[7], concavityFlag=concavity, timeFlag=time)
    

if example == 'Loan':
    if method == 'baseline':
        ex2.baseline(args[0], concavityFlag=concavity, timeFlag=time)
    elif method == 'random':
        ex2.inst_random(args[0], concavityFlag=concavity, timeFlag=time)
    elif method == "white-box":
        ex2.wb(args[0], concavityFlag=concavity, timeFlag=time)
    elif method == "grey-box-SAA":
        ex2.saa(args[0], int(args[1]), args[2], args[3], args[4], args[5], concavityFlag=concavity, timeFlag=time)
    elif method == "grey-box-SGA":
        print("This method is currently unavailable for Loan dataset.")
        quit(1)

if example == 'LGSSM':
    if method == 'baseline':
        ex3.baseline(args[0], args[1], concavityFlag=concavity, timeFlag=time)
    elif method == 'random':
        ex3.inst_random(args[0], args[1], concavityFlag=concavity, timeFlag=time)
    elif method == "white-box":
        ex3.wb(args[0], args[1], concavityFlag=concavity, timeFlag=time)
    elif method == "grey-box-SAA":
        ex3.saa(args[0], int(args[1]), args[2], concavityFlag=concavity, timeFlag=time)
    elif method == "grey-box-SGA":
        print("This method is currently unavailable for LGSSM dataset.")
        quit(1)
