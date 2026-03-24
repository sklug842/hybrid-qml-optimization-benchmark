#####
# Testing Gradient-Free Optimizers
#####


#####
# Python Libraries
#####

import sys
import os
import optuna
import numpy                               as     np
from   datetime                            import datetime
import tensorflow                          as     tf

from help_classes.wrapper_hybrid_qml_model import Wrapper_Hybrid_QML_Model

from gf_optimizers.ga_tourn                import GA_Tourn
from gf_optimizers.bb_bc                   import BB_BC
from gf_optimizers.ga_elite                import GA_Elite
from gf_optimizers.cem                     import CEM
from gf_optimizers.gwo                     import GWO
from gf_optimizers.aro                     import ARO
from gf_optimizers.nes                     import NES
from gf_optimizers.pso_bgb                 import PSO_BGB
from gf_optimizers.spo                     import SPO


#####
# Environment
#####

# Suppress most TensorFlow log messages (errors still shown)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Select which GPU to use (index passed as first CLI argument)
os.environ["CUDA_VISIBLE_DEVICES"] = str(int(sys.argv[1]))

# Enable dynamic GPU memory growth (prevents TF from pre-allocating all VRAM)
physical_gpus = tf.config.list_physical_devices("GPU")
if physical_gpus:
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Enumerate logical GPUs after configuring memory growth
logical_gpus = tf.config.list_logical_devices("GPU")
print("Logical GPUs:", logical_gpus)

# Use float64 throughout Keras (important for some quantum simulations)
tf.keras.backend.set_floatx("float64")


#####
# Usage
#####

# CLI:
# python3 test_optimizers_GF.py <CUDA_VISIBLE_DEVICES> <dataset> <type_QVC> <algorithm>
#
# Example:
# python3 test_optimizers_GF.py -1 "adult" "V1" "ARO"


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
# "ARO"
# "BB-BC"
# "CEM"
# "GA-Elite"
# "GA-Tourn"
# "GWO"
# "NES"
# "PSO"
# "PSO-BGB"
# "SPO"


#####
# CLI Arguments
#####

# Dataset name passed as second CLI argument
dataset           = str(sys.argv[2])

# QVC type passed as third CLI argument
type_QVC          = str(sys.argv[3])

# Optimizer name passed as fourth CLI argument
algorithm         = str(sys.argv[4])


#####
# Experiment Configuration
#####

# Number of independent runs per hyperparameter setting
n_runs            = 20

# Number of epochs logged per run
n_epochs          = 30

# Evaluation budget per epoch (used by the GF optimizers)
n_evals_per_epoch = 49


#####
# Optuna Trial Budget
#####

if   algorithm == "ARO":
    n_trials = 30

elif algorithm == "BB-BC":
    n_trials = 5**4

elif algorithm == "CEM":
    n_trials = 5**4*3

elif algorithm == "GA-Elite":
    n_trials = 5**5

elif algorithm == "GA-Tourn":
    n_trials = 5**6

elif algorithm == "GWO":
    n_trials = 5**4

elif algorithm == "NES":
    n_trials = 5**5

elif algorithm == "PSO":
    n_trials = 5**4

elif algorithm == "PSO-BGB":
    n_trials = 5**5

elif algorithm == "SPO":
    n_trials = 5**5


def objective(trial):
    """
    Optuna objective function.

    Returns a conservative performance estimate (mean - 3*std) at the best epoch,
    computed across `n_runs` independent runs for the sampled hyperparameters.
    """

    if algorithm == "ARO":

        sigma_init = trial.suggest_float("sigma_init", 0.1, 2.2, step=0.4)
        sigma_gs   = trial.suggest_float("sigma_gs"  , 0.1, 1.2, step=0.2)


    elif algorithm == "BB-BC":

        pop_size   = trial.suggest_int(  "pop_size",  4        ,  13       , step=2        )
        alpha      = trial.suggest_float("alpha"   ,  0.1      ,  2.2      , step=0.2      )
        x_min      = trial.suggest_float("x_min"   , -1.5*np.pi, -0.5*np.pi, step=0.5*np.pi)
        x_max      = trial.suggest_float("x_max"   ,  0.5*np.pi,  1.5*np.pi, step=0.5*np.pi)


    elif algorithm == "CEM":

        pop_size   = trial.suggest_int(  "pop_size"  ,   4      ,  13      , step=2        )
        best_size  = trial.suggest_int(  "best_size" ,   2      ,  11      , step=2        )
        mu_init    = trial.suggest_float("mu_init"   ,  -1      , 1.1      , step=0.5      )
        sigma_init = trial.suggest_float("sigma_init", 0.1      , 1.8      , step=0.4      )
        beta       = trial.suggest_float("beta"      , 0.2      , 1.1      , step=0.2      )
        x_min_max  = trial.suggest_float("x_min_max" , 0.5*np.pi, 1.5*np.pi, step=0.5*np.pi)

        if best_size > pop_size:
            raise optuna.exceptions.TrialPruned()


    elif algorithm == "GA-Elite":

        sigma_init     = trial.suggest_float("sigma_init"    , 0.1, 2.2, step=0.4)
        pop_size       = trial.suggest_int(  "pop_size"      ,   4,  13, step=2  )
        offspring_size = trial.suggest_int(  "offspring_size",   2,  11, step=2  )
        p_cx           = trial.suggest_float("p_cx"          , 0.1, 1.2, step=0.2)
        p_mut          = trial.suggest_float("p_mut"         , 0.1, 1.2, step=0.2)
        sigma_gs       = trial.suggest_float("sigma_gs"      , 0.1, 1.2, step=0.2)

        if offspring_size > pop_size:
            raise optuna.exceptions.TrialPruned()


    elif algorithm == "GA-Tourn":

        pop_size   = trial.suggest_int(  "pop_size"  ,   4,  13, step=2  )
        t_size     = trial.suggest_int(  "t_size"    ,   2,  11, step=2  )
        sigma_init = trial.suggest_float("sigma_init", 0.1, 2.2, step=0.4)
        p_mut      = trial.suggest_float("p_mut"     , 0.1, 1.2, step=0.2)
        p_cx       = trial.suggest_float("p_cx"      , 0.1, 1.2, step=0.2)
        sigma_gs   = trial.suggest_float("sigma_gs"  , 0.1, 1.2, step=0.2)

        if t_size > pop_size:
            raise optuna.exceptions.TrialPruned()


    elif algorithm == "GWO":

        pop_size = trial.suggest_int(  "pop_size",          4,         13, step=2  )
        x_min    = trial.suggest_float("x_min"   , -2.0*np.pi, -0.5*np.pi, step=0.5)
        x_max    = trial.suggest_float("x_max"   ,  0.5*np.pi,  2.0*np.pi, step=0.5)


    elif algorithm == "NES":

        pop_size   = trial.suggest_int(  "pop_size"  ,   4, 13 , step=2  )
        mu_eta     = trial.suggest_float("mu_eta"    , 0.1, 1.1, step=0.2)
        sigma_eta  = trial.suggest_float("sigma_eta" , 0.1, 1.1, step=0.2)
        mu_init    = trial.suggest_float("mu_init"   ,  -1, 1.1, step=0.5)
        sigma_init = trial.suggest_float("sigma_init", 0.2, 1.5, step=0.2)


    elif algorithm.split("-")[0] == "PSO":

        pop_size = trial.suggest_int(  "pop_size",          4,         13, step=2  )
        x_min    = trial.suggest_float("x_min"   , -1.5*np.pi, -0.3*np.pi, step=0.5)
        x_max    = trial.suggest_float("x_max"   ,  0.5*np.pi,  1.7*np.pi, step=0.5)
        w        = trial.suggest_float("w"       ,        0.1,       0.99, step=0.1)
        c1       = trial.suggest_float("c1"      ,        0.5,       3.10, step=0.5)
        c2       = trial.suggest_float("c2"      ,        0.5,       3.10, step=0.5)

        if algorithm == "PSO":
            flag_bgb = False

        elif algorithm == "PSO-BGB":
            flag_bgb        = True
            use_global_best = trial.suggest_int("use_global_best", 0, 1, step=1)


    elif algorithm == "SPO":

        pop_size = trial.suggest_int(  "pop_size"  ,          4,         13, step=2  )
        x_min    = trial.suggest_float("x_min"     , -1.5*np.pi, -0.5*np.pi, step=0.5)
        x_max    = trial.suggest_float("x_max"     ,  0.5*np.pi,  1.5*np.pi, step=0.5)
        d_root   = trial.suggest_float("d_root"    ,       0.01,       0.51, step=0.1)
        d_runner = trial.suggest_float("d_runner"  ,       0.50,       1.01, step=0.1)


    # Store metadata in the trial for later inspection
    trial.set_user_attr("algorithm", algorithm)
    trial.set_user_attr("type_QVC" , type_QVC )
    trial.set_user_attr("dataset"  , dataset  )
    trial.set_user_attr("n_runs"   , n_runs   )


    # Collect per-epoch test accuracies for each run
    accuracies = np.zeros((n_epochs, n_runs))


    # Repeat training multiple times for robustness (stochasticity)
    for run in range(n_runs):

        myWrapper = Wrapper_Hybrid_QML_Model(dataset     = dataset,
                                             type_QVC    = type_QVC,
                                             n_qubits    = 3,
                                             n_layers    = 2,
                                             n_shots     = 10000,
                                             max_threads = 50)


        if algorithm == "ARO":

            accuracies[:, run] = ARO(wrapper           = myWrapper,
                                     n_evals_per_epoch = n_evals_per_epoch,
                                     n_epochs          = n_epochs,
                                     sigma_init        = sigma_init,
                                     sigma_gs          = sigma_gs).run()


        elif algorithm == "BB-BC":

            accuracies[:, run] = BB_BC(wrapper           = myWrapper,
                                       n_evals_per_epoch = n_evals_per_epoch,
                                       n_epochs          = n_epochs,
                                       pop_size          = pop_size,
                                       alpha             = alpha,
                                       x_min             = x_min,
                                       x_max             = x_max).run()


        elif algorithm == "CEM":

            accuracies[:, run] = CEM(wrapper           = myWrapper,
                                     n_evals_per_epoch = n_evals_per_epoch,
                                     n_epochs          = n_epochs,
                                     pop_size          = pop_size,
                                     mu_init           = mu_init,
                                     sigma_init        = sigma_init,
                                     best_size         = best_size,
                                     beta              = beta,
                                     x_min_max         = x_min_max).run()


        elif algorithm == "GA-Elite":

            accuracies[:, run] = GA_Elite(wrapper           = myWrapper,
                                          n_evals_per_epoch = n_evals_per_epoch,
                                          n_epochs          = n_epochs,
                                          sigma_init        = sigma_init,
                                          pop_size          = pop_size,
                                          offspring_size    = offspring_size,
                                          p_cx              = p_cx,
                                          p_mut             = p_mut,
                                          sigma_gs          = sigma_gs).run()


        elif algorithm == "GA-Tourn":

            accuracies[:, run] = GA_Tourn(wrapper           = myWrapper,
                                          n_evals_per_epoch = n_evals_per_epoch,
                                          n_epochs          = n_epochs,
                                          sigma_init        = sigma_init,
                                          pop_size          = pop_size,
                                          t_size            = t_size,
                                          p_mut             = p_mut,
                                          p_cx              = p_cx,
                                          sigma_gs          = sigma_gs).run()


        elif algorithm == "GWO":

            accuracies[:, run] = GWO(wrapper           = myWrapper,
                                     n_evals_per_epoch = n_evals_per_epoch,
                                     n_epochs          = n_epochs,
                                     pop_size          = pop_size,
                                     x_min             = x_min,
                                     x_max             = x_max).run()


        elif algorithm == "NES":

            accuracies[:, run] = NES(wrapper           = myWrapper,
                                     n_evals_per_epoch = n_evals_per_epoch,
                                     n_epochs          = n_epochs,
                                     pop_size          = pop_size,
                                     mu_eta            = mu_eta,
                                     sigma_eta         = sigma_eta,
                                     mu_init           = mu_init,
                                     sigma_init        = sigma_init).run()


        elif algorithm == "PSO":

            accuracies[:, run] = PSO_BGB(wrapper           = myWrapper,
                                         n_evals_per_epoch = n_evals_per_epoch,
                                         n_epochs          = n_epochs,
                                         pop_size          = pop_size,
                                         x_min             = x_min,
                                         x_max             = x_max,
                                         w                 = w,
                                         c1                = c1,
                                         c2                = c2,
                                         flag_bgb          = flag_bgb).run()


        elif algorithm == "PSO-BGB":

            accuracies[:, run] = PSO_BGB(wrapper           = myWrapper,
                                         n_evals_per_epoch = n_evals_per_epoch,
                                         n_epochs          = n_epochs,
                                         pop_size          = pop_size,
                                         x_min             = x_min,
                                         x_max             = x_max,
                                         w                 = w,
                                         c1                = c1,
                                         c2                = c2,
                                         flag_bgb          = flag_bgb,
                                         use_global_best   = use_global_best).run()


        elif algorithm == "SPO":

            accuracies[:, run] = SPO(wrapper           = myWrapper,
                                     n_evals_per_epoch = n_evals_per_epoch,
                                     n_epochs          = n_epochs,
                                     pop_size          = pop_size,
                                     x_min             = x_min,
                                     x_max             = x_max,
                                     d_root            = d_root,
                                     d_runner          = d_runner).run()


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


    # Objective value to maximize (best conservative epoch score)
    return float(accuracies_adjusted[best_n_epoch])



if __name__ == "__main__":

    # Record start time for logging
    start_berechnung = datetime.now()

    # Compose unique Optuna study name
    study_name       = "GF_" + algorithm.replace(" ", "_") + "_" + dataset.replace(" ", "_") + "_" + type_QVC

    # Create / load Optuna study
    study            = optuna.create_study(study_name     = study_name,
                                           sampler        = optuna.samplers.CmaEsSampler(),
                                           direction      = "maximize",
                                           storage        = "sqlite:///../optuna/" + dataset.replace(" ", "_") + "/" + study_name + ".db",
                                           load_if_exists = True)

    # Optimize the objective defined above
    study.optimize(func     = objective,
                   n_trials = n_trials)

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