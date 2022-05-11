import argparse
import json
import os
from steams.train_eval_pred.k_fold import k_fold_eval_function
from steams.utils.utils import get_random_alphanumeric_string

def main():
    """
    Main k_fold_eval function.
    """
    parser = argparse.ArgumentParser(description="Argument parsing for k-fold evaluation")
    parser.add_argument("-w", help="path of the workdir")
    parser.add_argument("-c", help="json configuration file providing the whole set of parameter")
    parser.add_argument("-s", help="json sessions file providing keys about outcomes from aqdl modules")
    args = parser.parse_args()
    if not os.path.exists(args.w):
        raise ValueError("path "+args.w+" does not exist")
    with open(os.path.join(args.w, "config",args.c),encoding='utf8') as f:
        get_steams_config = json.load(f)
    with open(os.path.join(args.w, "config",args.s),encoding='utf8') as f:
        get_session_config = json.load(f)
    # create a new session for results
    key = get_random_alphanumeric_string(20)
    get_steams_config['sessiondir'] = os.path.join(args.w,"session",key)
    os.makedirs(get_steams_config['sessiondir'], exist_ok=True)
    # require data from steams_train
    get_steams_config['training_param']["model_dir"] = os.path.join(args.w,'session',get_session_config['steams_train'][0])
    # require data from collosm
    get_steams_config["data"]["path"] = os.path.join(args.w,'session',get_session_config['data'][0])
    k_fold_eval_function(get_steams_config)
    print("k-fold evaluation completed.")
    #save session of the results in sessions file
    get_session_config['steams_eval']=[key]
    with open(os.path.join(args.w, "config",args.s), 'w',encoding='utf8') as f_sessions:
        json.dump(get_session_config, f_sessions)

if __name__ == "__main__":
    main()
