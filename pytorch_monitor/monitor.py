import socket
import datetime
import sh
import os
import random

import torch
import torch.autograd as ag
from torchvision.utils import make_grid
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter

def grad_hook(module, name, writer, bins):
    def hook(grad):
        writer.add_histogram('{}/grad'.format(name.replace('.','/')),
                             grad.data,
                             module.global_step-1,
                             bins=bins)
    return hook

def remove_old_var_hooks(module, input):
    """ Removes all old registered intermeditate variable hooks from the module
    before applying them on the forward pass, so stale closures don't happen.
    """
    for hook in list(module.var_hooks.keys()):
        module.var_hooks[hook].remove()
        del module.var_hooks[hook]
    
def monitor_module(module, summary_writer,
                   track_data=True,
                   track_grad=True,
                   track_update=True,
                   track_update_ratio=False, # this is usually unnecessary
                   bins=51):
    """ Allows for remote monitoring of a module's params and buffers.
    The following may be monitored:
      1. Forward Values - Histograms of the values for parameter and buffer tensors
      2. Gradient Values - Histograms of the gradients for parameter and buffer tensors
      3. Update Values - Histograms of the change in parameter and buffer tensor
           values from the last recorded forward pass
      4. Update Ratios - Histograms of the ratio of change in value for parameter
           and value tensors from the last iteration to the actual values.
           I.e., what is the relative size of the update.
           Generally we like to see values of about .001.
           See [cite Andrej Karpathy's babysitting dnn's blog post]
    """
    def monitor_forward_and_vars(module, input, output):
        # iterate over the state after the forward pass
        # registering backprop hooks on intermediate variables
        # (allowing us to not need to retain grads)
        # as well as recording the forward prop activations
        # and the updates from the last iteration if possible
        for name, tensor in module.state_dict().items():
            if isinstance(tensor, ag.Variable): # it's an intermediate computation
                if track_grad:
                    hook = grad_hook(module, name, summary_writer, bins)
                    module.var_hooks[name] = tensor.register_hook(hook)
                if track_data:
                    summary_writer.add_histogram('{}/data'.format(name.replace('.','/')),
                                                 tensor.data,
                                                 module.global_step,
                                                 bins=bins)

            else: # it's a param tensor
                if track_data:
                    summary_writer.add_histogram('{}/data'.format(name.replace('.','/')),
                                                 tensor,
                                                 module.global_step,
                                                 bins=bins)
                if name in module.last_state_dict:
                    if track_update:
                        update = tensor - module.last_state_dict[name]
                        summary_writer.add_histogram('{}/update-val'.format(name.replace('.','/')),
                                                     update,
                                                     module.global_step-1,
                                                     bins=bins)
                    if track_update and track_update_ratio:
                        update_ratio = update / (module.last_state_dict[name]+1e-15)

                        summary_writer.add_histogram('{}/update-ratio'.format(name.replace('.','/')),
                                                     update_ratio,
                                                     module.global_step-1,
                                                     bins=bins)
                module.last_state_dict[name] = tensor.clone()
        module.global_step += 1

    if not hasattr(module, 'global_step'):
        module.global_step = 0
    if not hasattr(module, 'last_state_dict'):
        module.last_state_dict = dict()
    if not hasattr(module, 'var_hooks'):
        module.var_hooks = dict()
    if not hasattr(module, 'param_hooks'):
        module.param_hooks = dict()

    # monitor the backward grads for params
    if track_grad:
        param_names = [ name for name, _ in module.named_parameters()]
        for name, param in zip(param_names, module.parameters()):
            hook = grad_hook(module, name, summary_writer, bins)
            module.param_hooks[name] = param.register_hook(hook)
            
    # monitor forward grads
    module.register_forward_pre_hook(remove_old_var_hooks)
    module.register_forward_hook(monitor_forward_and_vars)
    
def commit(experiment_name, time):
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
    text += 'Commit Hash: {}\n'.format(commit(config.get('title', '<No Title'), start_time))
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
    if rseed:
        random.seed(rseed)
        torch.manual_seed(rseed)

    writer.add_text(config['tag'], text, 0)
    
    # save the config to run dir
    with open(os.path.join(config['run_dir'], 'config.json'), 'w') as f:
        json.dumps(config, f, indent=2)
        
    return writer, config
