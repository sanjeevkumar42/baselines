import json
import os


def int_value(x):
    value = -1
    try:
        value = int(x.strip())
    except:
        pass
    return value


def get_exp_dir(logdir, env_id, dir_prefix):
    """ Get name of next available dir."""
    outdir = os.path.join(logdir, env_id)
    os.makedirs(outdir, exist_ok=True)
    exp_file_ids = []
    for f in os.listdir(outdir):
        expdir = os.path.join(outdir, f)
        if f.startswith(dir_prefix + '_'):# and any([d.startswith('model.ckpt') for d in os.listdir(expdir)]):
            exp_file_ids.append(int_value(f.split('_')[1]))

    if len(exp_file_ids) > 0:
        exp_dir = '{}_{}'.format(dir_prefix, max(exp_file_ids) + 1)
    else:
        exp_dir = dir_prefix + '_0'
    exp_dir = os.path.join(logdir, env_id, exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_params(expdir, params):
    params_file = os.path.join(expdir, 'params.json')
    params_json = json.dumps(params, sort_keys=True, indent=4)
    fw = open(params_file, 'w')
    fw.write(params_json)


# maybe move to decorator?
def prepare_exp_dirs(script_params, outdir, env_id, dir_prefix='exp'):
    exp_dir = get_exp_dir(outdir, env_id, dir_prefix)
    save_params(exp_dir, script_params)
    return exp_dir
