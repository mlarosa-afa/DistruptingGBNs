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
            3. grey-box-SGA--u_1, method, Psi coefficent, mu coefficent, Kappa, nu, learn rate, epsilon
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
        print("0.1.0")
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

if args[1] in ["white-box", "grey-box-SAA", "grey-box-SGA"]:
    method = args[1]
else:
    raise Exception(
        "method argument passed with invalid option. Valid options: 'white-box', 'grey-box-SAA', and 'grey-box-SGA'")

#Ensures all parameters passed
if example == "LGSSM" and method == "white-box" and not (len(args) == 4):
    raise Exception(
        "Arguments passed to 'white-box' for 'LGSSM' are incorrect. Must include only: u_1, risk_tolerance")
elif not(example=="LGSSM") and method == "white-box" and not (len(args) == 3):
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
    print("SOLUTION:")
    if method == "white-box":
        print("U_1\tU_2\tZ_0\tZ_1\tZ_2\tZ_3\tZ_4\tZ_5\tZ_6\tZ_7\tZ_8\tZ_9\tZ_10\tobj_val\tphi_opt1\tphi_opt2")
        ex1.zillow_wb(args[0], concavityFlag=concavity, timeFlag=time)
    elif method == "grey-box-SAA":
        print("U_1\tU_2\tZ_0\tZ_1\tZ_2\tZ_3\tZ_4\tZ_5\tZ_6\tZ_7\tZ_8\tZ_9\tZ_10\tobj_val\tphi_opt1\tphi_opt2")
        ex1.zillow_saa(args[0], int(args[1]), args[2], args[3], args[4], args[5], timeFlag=time)
    elif method == "grey-box-SGA":
        print("U_1\tU_2\tZ_0\tZ_1\tZ_2\tZ_3\tZ_4\tZ_5\tZ_6\tZ_7\tZ_8\tZ_9\tZ_10\tobj_val")
        ex1.zillow_sgd(args[0], args[2], args[3], args[4], args[5], int(args[1]), args[6], args[7], timeFlag=time)

if example == 'Loan':
    if method == "white-box":
        print("SOLUTION:")
        print("U_1\tU_2\tZ_0\tZ_1\tZ_2\tZ_3\tZ_4\tZ_5\tZ_6\tobj_val\tphi_opt1\tphi_opt2")
        ex2.loan_wb(args[0], concavityFlag=concavity, timeFlag=time)
    elif method == "grey-box-SAA":
        print("SOLUTION:")
        print("U_1\tU_2\tZ_0\tZ_1\tZ_2\tZ_3\tZ_4\tZ_5\tZ_6\tobj_val\tphi_opt1\tphi_opt2")
        ex2.loan_saa(args[0], int(args[1]), args[2], args[3], args[4], args[5], timeFlag=time)
    elif method == "grey-box-SGA":
        print("This method is currently unavailable for Loan data set.")
        quit(1)

if example == 'LGSSM':
    if method == "white-box":
        print("SOLUTION:")
        print("U_1\tU_2\tZ_0\tZ_1\tZ_2\tZ_3\tZ_4\tZ_5\tZ_6\tZ_7\tZ_8\tZ_9\tZ_10"
              "\tZ_11\tZ_12\tZ_13\tZ_14\tZ_15\tZ_16\tZ_17\tZ_18\tZ_19\tZ_20\tobj_val\tphi_opt1\tphi_opt2")
        ex3.lgssm_wb(args[0], args[1], concavityFlag=concavity, timeFlag=time)
    elif method == "grey-box-SAA":
        print("SOLUTION:")
        print("U_1\tU_2\tZ_0\tZ_1\tZ_2\tZ_3\tZ_4\tZ_5\tZ_6\tZ_7\tZ_8\tZ_9\tZ_10"
              "\tZ_11\tZ_12\tZ_13\tZ_14\tZ_15\tZ_16\tZ_17\tZ_18\tZ_19\tZ_20\tobj_val\tphi_opt1\tphi_opt2")
        ex3.lgssm_saa(args[0], int(args[1]), args[2],  timeFlag=time)
    elif method == "grey-box-SGA":
        print("This method is currently unavailable for LGSSM data set.")
        quit(1)
