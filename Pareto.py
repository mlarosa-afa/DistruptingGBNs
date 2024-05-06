import sys
from numpy import arange
import numpy as np

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

    if example == 'Zillow':
        import Example1_Zillow as ex
    elif example == 'Loan':
        import Example2_Loans as ex
    elif example == 'LGSSM':
        import Example3_LGSSM as ex

    results = []
    for i in arange(0.0, 1.01, 0.05):
        if method == "baseline":
            _, phi_1, phi_2 = ex.baseline(i, verbose=False)
        elif method == "random":
            _, _, phi_1, phi_2 = ex.inst_random(i, verbose=False)
        elif method == "white-box":
            _, _, phi_1, phi_2 = ex.wb(i, verbose=False)
        elif method == "grey-box-SAA":
            _, _, phi_1, phi_2 = ex.saa(i, int(args[1]), args[2], args[3], args[4], args[5], verbose=False)
        elif method == "grey-box-SGD":
            _, _, phi_1, phi_2 = ex.sgd(i, int(args[1]), args[2], args[3], args[4], args[5], args[6], verbose=False)
        results.append([i, phi_1, phi_2])
    
    np.savetxt(filename, np.asarray(results), delimiter="\t")