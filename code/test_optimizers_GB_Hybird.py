######
# Testing Gradient-Based and Hybrid Gradient-Based Optimizers
######


#####
# Python Libraries
#####

import sys
import os
import optuna
import numpy                                 as     np
from   datetime                              import datetime

import gc
from   tensorflow.keras                      import backend as K

from   help_classes.wrapper_hybrid_qml_model import Wrapper_Hybrid_QML_Model
from   gb_hybrid_optimizers.adam_sgd         import Adam_SGD
from   gb_hybrid_optimizers.gls_adam_sgd     import GLS_Adam_SGD


#####
# Environment
#####

# Suppress most TensorFlow log messages (errors still shown)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Select which GPU to use (index passed as first CLI argument)
os.environ["CUDA_VISIBLE_DEVICES"] = str(int(sys.argv[1]))


#####
# Usage
#####

# CLI:
# python3 test_optimizers_GB.py <CUDA_VISIBLE_DEVICES> <dataset> <type_QVC> <algorithm>
#
# Example:
# python3 test_optimizers_GB.py -1 "adult" "V1" "Adam"


#####
# Options
#####

# Datasets:
# "adult"
# "breast cancer"
# "heart disease"
# "network"

# Type QVCs:
# "V1"
# "V2"

# Algorithms:
# "Adam"
# "SGD"
# "GLS-Adam"
# "GLS-SGD"


#####
# CLI Arguments
#####

# Dataset name passed as second CLI argument
dataset   = str(sys.argv[2])

# QVC type passed as third CLI argument ("V1" or "V2")
type_QVC  = str(sys.argv[3])

# Algorithm name passed as fourth CLI argument
algorithm = str(sys.argv[4])


#####
# Experiment Configuration
#####

# Number of independent runs per hyperparameter setting
n_runs            = 20

# Number of training epochs per run
n_epochs          = 30

# Nominal evaluations per epoch (used by GLS variant)
n_evals_per_epoch = 49


#####
# Optuna Trial Budget
#####

if   algorithm == "Adam":
    n_trials = 5**3

elif algorithm == "SGD":
    n_trials = 60

elif algorithm == "GLS-Adam":
    n_trials = 5**4

elif algorithm == "GLS-SGD":
    n_trials = 60*4


def objective(trial):
    """
    Optuna objective function.

    Returns a conservative performance estimate (mean - 3*std) at the best epoch,
    computed across `n_runs` independent runs for the sampled hyperparameters.
    """

    # Learning rate search (log-uniform)
    eta = trial.suggest_float("eta", 1e-5, 1e-1, log=True)

    if algorithm == "Adam" or algorithm == "GLS-Adam":

        # Underlying TF optimizer type
        optimizer = "Adam"

        # Adam beta1 search (coarse grid)
        beta_1    = trial.suggest_float("beta_1", 0.7400, 1.00, step=0.05)

        # Adam beta2 search (finer grid)
        beta_2    = trial.suggest_float("beta_2", 0.9499, 1.00, step=0.01)


    elif algorithm == "SGD" or algorithm == "GLS-SGD":

        # Underlying TF optimizer type
        optimizer = "SGD"

        # Momentum search (coarse grid)
        momentum  = trial.suggest_float("momentum", 0.09, 1.00, step=0.10)


    if algorithm == "GLS-Adam":

        # Placeholder: GLS Adam doesn't use momentum
        momentum  = -1


    elif algorithm == "GLS-SGD":

        # Placeholder: GLS SGD doesn't use beta_1 / beta_2
        beta_1    = -1
        beta_2    = -1


    if algorithm.split("-")[0] == "GLS":

        # GLS: number of offspring/copies per epoch (even-ish range)
        n_copies  = trial.suggest_int("n_copies", 4, 11, step=2)


    # Store metadata in the trial for later inspection
    trial.set_user_attr("algorithm", algorithm)
    trial.set_user_attr("type_QVC" , type_QVC )
    trial.set_user_attr("dataset"  , dataset  )
    trial.set_user_attr("n_runs"   , n_runs   )


    # Collect per-epoch test accuracies for each run
    accuracies = np.zeros((n_epochs, n_runs))


    # Repeat training multiple times for robustness (stochasticity)
    for run in range(n_runs):

        myWrapper = Wrapper_Hybrid_QML_Model(dataset     = dataset ,
                                             type_QVC    = type_QVC,
                                             n_qubits    = 3       ,
                                             n_layers    = 2       ,
                                             n_shots     = 10000   ,
                                             max_threads = 50      )


        if algorithm == "Adam":

            accuracies[:, run] = Adam_SGD(wrapper   = myWrapper,
                                          n_epochs  = n_epochs ,
                                          optimizer = optimizer,
                                          eta       = eta      ,
                                          beta_1    = beta_1   ,
                                          beta_2    = beta_2   ).run()


        elif algorithm == "SGD":

            accuracies[:, run] = Adam_SGD(wrapper   = myWrapper,
                                          n_epochs  = n_epochs ,
                                          optimizer = optimizer,
                                          eta       = eta      ,
                                          momentum  = momentum ).run()


        elif algorithm.split("-")[0] == "GLS":

            accuracies[:, run] = GLS_Adam_SGD(wrapper           = myWrapper        ,
                                              n_evals_per_epoch = n_evals_per_epoch,
                                              n_epochs          = n_epochs         ,
                                              n_copies          = n_copies         ,
                                              optimizer         = optimizer        ,
                                              eta               = eta              ,
                                              momentum          = momentum         ,
                                              beta_1            = beta_1           ,
                                              beta_2            = beta_2           ).run()


    # Mean accuracy per epoch across runs
    accuracies_mean     = np.mean(accuracies, axis=1)

    # Std deviation per epoch across runs
    accuracies_std      = np.std(accuracies, axis=1)

    # Conservative score (mean - 3σ) to penalize instability
    accuracies_adjusted = accuracies_mean - 3 * accuracies_std

    # Pick epoch maximizing conservative score
    best_n_epoch        = int(np.argmax(accuracies_adjusted))


    # Store best epoch (1-indexed) and summary stats
    trial.set_user_attr("best_n_epoch"     , best_n_epoch + 1)
    trial.set_user_attr("best_n_epoch_mean", float(accuracies_mean[best_n_epoch]))
    trial.set_user_attr("best_n_epoch_std" , float(accuracies_std[best_n_epoch]))


    # Store per-epoch stats and raw runs
    for x in range(n_epochs):
        trial.set_user_attr("mean_" + str(x), float(accuracies_mean[x]))

    for x in range(n_epochs):
        trial.set_user_attr("std_" + str(x), float(accuracies_std[x]))

    for x in range(n_epochs):
        trial.set_user_attr("accuracies_" + str(x), accuracies[x].tolist())


    # Clear TF/Keras state (helps prevent graph/memory buildup)
    K.clear_session()

    # Force Python garbage collection to free memory
    gc.collect()

    # Objective value to maximize (best conservative epoch score)
    return float(accuracies_adjusted[best_n_epoch])



if __name__ == "__main__":

    # Record start time for logging
    start_berechnung = datetime.now()

    # Compose unique Optuna study name
    study_name       = "GB_" + algorithm.replace(" ", "_") + "_" + dataset.replace(" ", "_") + "_" + type_QVC

    # Create / load Optuna study
    study            = optuna.create_study(study_name     = study_name                                                                ,
                                           sampler        = optuna.samplers.CmaEsSampler()                                            ,
                                           direction      = "maximize"                                                                ,
                                           storage        = "sqlite:///../optuna/" + dataset.replace(" ", "_") + "/" + study_name + ".db",
                                           load_if_exists = True                                                                      )

    # Optimize the objective defined above
    study.optimize(func     = objective,
                   n_trials = n_trials )

    # Record end time
    end_berechnung   = datetime.now()


    print(f" ")
    print(f"---------------------------------------------------------")
    print(f"Dataset:                {dataset                        }")
    print(f"Algorithm:              {algorithm                      }")
    print(f"Type of QVC:            {type_QVC                       }")
    print(f"Study Name:             {study_name                     }")
    print(f"# Runs per Combination: {n_runs                         }")
    print(f"# Trials:               {n_trials                       }")
    print(f"Start:                  {start_berechnung               }")
    print(f"End:                    {end_berechnung                 }")
    print(f"Duration:               {end_berechnung-start_berechnung}")
    print(f"---------------------------------------------------------")
    print(f" ")

    # Report how many trials were completed
    print("Number of finished trials: {}".format(len(study.trials)))

    # Retrieve best trial according to objective value
    trial = study.best_trial

    print("Best trial:")
    print("Value: {}".format(trial.value))

    print("Params:")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))