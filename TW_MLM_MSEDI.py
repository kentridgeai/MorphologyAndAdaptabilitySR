import warnings
import time
from genetic_modified import SymbolicRegressor
from fitness import make_fitness
from functions import make_function
from sklearn.utils.random import check_random_state
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
def NN_MSEDI(equation_str,lower_bound,upper_bound,num_points=20,resume=(),trials=1):
    
    data_list=list(resume)
    
    for rng_state in range(len(resume),trials):
        print(f"RNG_STATE:{rng_state}")
        start_time = time.time()
        rng = check_random_state(rng_state)

        # Training samples
        X_train = np.array(sorted(rng.uniform(lower_bound, upper_bound, num_points))).reshape(num_points, 1)
        y_train = eval(equation_str.replace("X","X_train[:, 0]"))

        # Testing samples
        X_test = np.linspace(lower_bound, upper_bound, num=10000).reshape(10000, 1)
        y_test = eval(equation_str.replace("X","X_test[:, 0]"))

        def _exp(x1):
            with np.errstate(over='ignore'):
                return np.where(np.abs(x1) < 100, np.exp(x1), 0.)
        exp = make_function(function=_exp,
                                name='exp',
                                arity=1)

        def _pow(x2, x1):
            with np.errstate(over='ignore'):
                x2 = np.abs(x2)
                x2 = np.where(x2 < 100, x2, 100)
                x1 = np.where(np.abs(x1)>0.01, x1, 0.01)
                ans = np.where(np.abs(x1) < 100, np.abs(x2)**x1, 0.)
                ans = np.where(np.abs(ans) < 10000, ans, 0.)
                return ans
        pow = make_function(function=_pow,
                                name='pow',
                                arity=2)

        def _arcsin(x1):
            x1 = np.where(x1 < 1, x1, 1)
            x1 = np.where(x1 > -1, x1, -1)
            return np.arcsin(x1)

        arcsin = make_function(function=_arcsin,
                                name='arcsin',
                                arity=1)

        def _msedi(y, y_pred, w):
            """Calculate the msedi."""
            return np.average(((np.diff(y_pred) - np.diff(y)) ** 2), weights=w[:-1])

        msedi = make_fitness(_msedi, greater_is_better=False)
        stopping_criteria=0.0000001
        est_gp = SymbolicRegressor(metric=msedi,population_size=40, init_depth = (2,4),
                                   generations=1, stopping_criteria=stopping_criteria,
                                   p_crossover=0.7, p_subtree_mutation=0.1,
                                   p_hoist_mutation=0.05, p_point_mutation=0.1,
                                   max_samples=0.9, verbose=1,
                                   parsimony_coefficient=0.01, random_state=rng_state,low_memory=True,
                                  function_set=['add', 'mul','sub','div','sin','cos','log',exp,pow,arcsin])

        est_gp.fit(X_train, y_train)

        for i in range(50):    
            est_gp.set_params(generations=(i+1)*5, metric="mse", warm_start=True)
            if i==0 or est_gp.run_details_["best_fitness"][-2]>stopping_criteria:
                est_gp.fit(X_train, y_train)
            else:
                break
            if i != 49:
                est_gp.set_params(generations=(i+1)*5+1, metric=msedi, warm_start=True)
                if est_gp.run_details_["best_fitness"][-1]>stopping_criteria:
                    est_gp.fit(X_train, y_train)
                else:
                    break
            if time.time()-start_time>600:
                break

        data_list.append(r2_score(y_test,est_gp.predict(X_train, y_train, X_test)))
        print(data_list)

    num_list=[]
    for i in data_list:
        num_list.append(i)
