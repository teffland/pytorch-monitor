import socket
import datetime
import sh
import os
import json

import random
import numpy.random as npr
import torch

from tensorboardX import SummaryWriter

def commit(experiment_name, time):
    """
    Try to commit repo exactly as it is when starting the experiment for reproducibility.
    """
    try:
        sh.git.commit('-a',
                m='"auto commit tracked files for new experiment: {} on {}"'.format(experiment_name, time),
                allow_empty=True
            )
        commit_hash = sh.git('rev-parse', 'HEAD').strip()
        return commit_hash
    except:
        return '<Unable to commit>'

def init_experiment(config):
    start_time = datetime.datetime.now().strftime('%b-%d-%y@%X')
    host_name = socket.gethostname()
    run_name = config.get('run_name', None)
    if run_name is None:
        run_name = '{}-{}'.format(start_time, host_name)
    run_comment = config.get('run_comment', None)
    if run_comment:
        run_name += '-{}'.format(run_comment)
    config['run_name'] = run_name

    # create the needed run directory ifnexists
    log_dir = config.get('log_dir', 'runs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    run_dir = os.path.join(log_dir, run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    config['run_dir'] = run_dir

    writer = SummaryWriter(run_dir)

    # commit if you can
    commit_hash = commit(config.get('title', '<No Title'), start_time)

    # create text summary for logging config to tensorboard
    config['tag'] = 'Experiment Config: {} :: {}\n'.format(
        config.get('title', '<No Title>'), start_time)

    text  = '<h3>{}</h3>\n'.format(config['tag'])
    text += '{}\n'.format(config.get('description', '<No Description>'))

    text += '<pre>'
    text += 'Start Time: {}\n'.format(start_time)
    text += 'Host Name: {}\n'.format(host_name)
    text += 'CWD: {}\n'.format(os.getcwd())
    text += 'PID: {}\n'.format(os.getpid())
    text += 'Log Dir: {}\n'.format(log_dir)
    text += 'Commit Hash: {}\n'.format(commit_hash)
    text += 'Random Seed: {}\n'.format(config.get('random_seed', '<Unknown...BAD PRACTICE!>'))
    text += '</pre>\n<pre>'

    skip_keys = ['tag', 'title', 'description', 'random_seed', 'log_dir', 'run_dir', 'run_name', 'run_comment']
    for key, val in config.items():
        if key in skip_keys:
            continue
        text += '{}: {}\n'.format(key, val)
    text += '</pre>'

    # set random seed
    rseed = config.get('random_seed', None)
    if rseed is not None:
        random.seed(rseed)
        npr.seed(rseed)
        torch.manual_seed(rseed)

    writer.add_text(config['tag'], text, 0)

    # save the config to run dir
    with open(os.path.join(config['run_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    return writer, config
