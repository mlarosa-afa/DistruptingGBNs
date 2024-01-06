import sys
from numpy import arange
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
    orig_stdout = sys.stdout
    f = open(filename, 'w+')
    sys.stdout = f
    if example == 'Zillow':
        print("U_1\tU_2\tZ_0\tZ_1\tZ_2\tZ_3\tZ_4\tZ_5\tZ_6\tZ_7\tZ_8\tZ_9\tZ_10\tobj_val\tphi_opt1\tphi_opt2")
    if example == 'Loan':
        print("U_1\tU_2\tZ_0\tZ_1\tZ_2\tZ_3\tZ_4\tZ_5\tZ_6\tobj_val\tphi_opt1\tphi_opt2")
    if example == 'LGSSM':
        print("U_1\tU_2\tZ_1\tZ_2\tZ_3\tZ_4\tZ_5\tZ_6\tZ_7\tZ_8\tZ_9\tZ_10"
              "\tZ_11\tZ_12\tZ_13\tZ_14\tZ_15\tZ_16\tZ_17\tZ_18\tZ_19\tZ_20\tobj_val\tphi_opt1\tphi_opt2")

    for i in arange(0.0, 1.01, 0.05):
        if example == 'Zillow':
            if method == "white-box":
                ex1.zillow_wb(i)
            elif method == "grey-box-SAA":
                ex1.zillow_saa(i, int(args[1]), args[2], args[3], args[4], args[5])
            elif method == "grey-box-SGD":
                ex1.zillow_sgd(i, int(args[1]), args[2], args[3], args[4], args[5], args[6])

        if example == 'Loan':
            if method == "white-box":
                ex2.loan_wb(i)
            elif method == "grey-box-SAA":
                ex2.loan_saa(i, int(args[1]), args[2], args[3], args[4], args[5])

        if example == 'LGSSM':
            if method == "white-box":
                ex3.lgssm_wb(i)
            elif method == "grey-box-SAA":
                ex3.lgssm_saa(i, int(args[1]), args[2], args[3], args[4], args[5])

    sys.stdout = orig_stdout
    f.close()