import numpy as np
import json
import pandas as pd
import itertools
import csv
import random
from datetime import datetime

def train(config):
    """
    :param config: dictionary  {'lr': 1e-4 ,'batch_size': 32 , 'norm' : 'l2', 'drop_out':0.6}
    :return: best test accuracy
    """
    print('Train with hyperpara :')
    print(json.dumps(config))
    return np.random.randint(0,100)

def create_cvs_offline_monitor(name = None):
    import csv

    # Create file and open connection
    out_file = 'random_search_trials.csv'
    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)

    # Write column names
    headers = ['score', 'hyperparameters', 'iteration']
    writer.writerow(headers)
    of_connection.close()


def train_1(lr , batch_size,norm,drop_out):
    print('Evaluate: lr {}  batch_size {} norm {} dropout {}'.format(lr,batch_size,norm,drop_out))
    return np.random.randint(0,100)

def get_objective(train):
    """ bridge from config to original train function funcion
    ==> We should reimplement this function to make a correct mapping
    """

    def objective(parameter):
        lr = parameter['lr']
        batch_size = parameter['batch_size']
        norm = parameter['norm']
        drop_out = parameter['drop_out']
        acc = train(lr, batch_size,norm,drop_out)
        return acc

    return objective

def grid_search(train, config, out_file ='grid_search.csv', max_evals = None):
    combi = 1
    for x in config.values():
        combi *= len(x)
    print('Total combination: {}'.format(combi))

    if max_evals is None:
        max_evals = combi
    elif max_evals > combi:
        max_evals = combi

    results = pd.DataFrame(columns = ['score','params','iteration'],index = list(range(max_evals)))
    param_grid = config
    keys, values = zip(*param_grid.items())
    i = 0
    # Write test time
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow(("====", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "===="))
    # make sure to close connection
    of_connection.close()

    for v in itertools.product(*values):
        selected_param = dict(zip(keys, v))
        # evaluate  the hyperparameters
        acc = train(selected_param)
        # store result
        results.loc[i,:] = (acc,selected_param,i)
        # open connection (append option) and write results
        of_connection = open(out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow((acc,selected_param,i))

        # make sure to close connection
        of_connection.close()
        i +=1
        if i > max_evals:
            break

    results.sort_values('score', ascending = False,inplace= True)
    results.reset_index(inplace=True)

    print('Best acc: {} with params {}'.format(results.loc[0,'score'],results.loc[0,'params']))
    return  results



def random_search(train, config, out_file ='random_search_res.csv', max_evals = 1):
    combi = 1
    for x in config.values():
        combi *= len(x)
    print('Total combination: {}'.format(combi))

    results = pd.DataFrame(columns = ['score','params','iteration'],index = list(range(max_evals)))
    param_grid = config

    # Write test time
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow(("====", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "===="))
    # make sure to close connection
    of_connection.close()

    for i in range(max_evals):
        selected_param = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        # evaluate  the hyperparameters
        acc = train(selected_param)
        # store result
        results.loc[i,:] = (acc,selected_param,i)
        # open connection (append option) and write results
        of_connection = open(out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow((acc,selected_param,i))

        # make sure to close connection
        of_connection.close()

    results.sort_values('score', ascending = False,inplace= True)
    results.reset_index(inplace=True)

    print('Best acc: {} with params {}'.format(results.loc[0,'score'],results.loc[0,'params']))
    return  results

if __name__ == '__main__':
    grid_search(get_objective(train_1), {'lr': [1e-4, 1e-3] , 'batch_size': [32] , 'norm' : ['l2'], 'drop_out':[0.6]})
    random_search(get_objective(train_1), {'lr': [1e-4, 1e-3], 'batch_size': [32], 'norm': ['l2'], 'drop_out': [0.6]}, max_evals=5)