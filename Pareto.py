import sys
from numpy import arange
import numpy as np
import Example1_Zillow as ex1
import Example2_Loans as ex2
import Example3_LGSSM as ex3

def generate_pareto(filename, example, method, args):
    """
        Runs *example* with *method* for u_1 values 0 to 1 at 0.01 increments.

        Parameter
        ---------
        filename : str
            file location on where to save data.
        example : str
            Example name to execute. Options 'Zillow', 'Loan', and 'LGSSM'
        method : str
            Which problem formulation to execute. Options 'white-box', 'grey-box-SAA', and 'grey-box-SGD'
            SGD only available for Zillow data set
        args : array
            Additional arguments required to run method

        Returns
        ---------
        file
            tsv with all testing results
    """

    results = []
    for i in arange(0.0, 1.01, 0.05):
        if example == 'Zillow':
            if method == "baseline":
                obj, phi_1, phi_2 = ex1.zillow_baseline(i, verbose=False)
            elif method == "random":
                obj, _, phi_1, phi_2 = ex1.zillow_random(i, verbose=False)
            elif method == "white-box":
                obj, _, phi_1, phi_2 = ex1.zillow_wb(i, verbose=False)
            elif method == "grey-box-SAA":
                obj, _, phi_1, phi_2 = ex1.zillow_saa(i, int(args[1]), args[2], args[3], args[4], args[5], verbose=False)
            elif method == "grey-box-SGD":
                obj, _, phi_1, phi_2 = ex1.zillow_sgd(i, int(args[1]), args[2], args[3], args[4], args[5], args[6], verbose=False)

        if example == 'Loan':
            if method == "baseline":
                obj, phi_1, phi_2 = ex1.loan_baseline(i, verbose=False)
            elif method == "random":
                obj, _, phi_1, phi_2 = ex1.loan_random(i, verbose=False)
            elif method == "white-box":
                obj, _, phi_1, phi_2 = ex2.loan_wb(i, verbose=False)
            elif method == "grey-box-SAA":
                obj, _, phi_1, phi_2 = ex2.loan_saa(i, int(args[1]), args[2], args[3], args[4], args[5], verbose=False)

        if example == 'LGSSM':
            if method == "baseline":
                obj, phi_1, phi_2 = ex3.lgssm_baseline(i, verbose=False)
            elif method == "random":
                obj, _, phi_1, phi_2 = ex3.lgssm_random(i, verbose=False)
            elif method == "white-box":
                obj, _, phi_1, phi_2 = ex3.lgssm_wb(i, verbose=False)
            elif method == "grey-box-SAA":
                obj, _, phi_1, phi_2 = ex3.lgssm_saa(i, int(args[1]), args[2], args[3], args[4], args[5], verbose=False)
        results.append([i, phi_1, phi_2])
    
    np.savetxt(filename, np.asarray(results), delimiter="\t")

    #orig_stdout = sys.stdout
    #f = open(filename, 'w+')
    #sys.stdout = f
    #sys.stdout = orig_stdout
    #f.close()