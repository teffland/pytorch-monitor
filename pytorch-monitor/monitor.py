import socket
import datetime
import sh
import os
import random

import torch
import torch.autograd as ag
from torchvision.utils import make_grid
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


# SYMMETRIC_LOGSCALE_BINS = [-1e15, -1e5, -1e4, -1e3, -1e2, -1e1, -1e-1, -1e-2, -1e-3, -1e-4,
#       1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e15]

def grad_hook(module, name, writer):
    def hook(grad):
        writer.add_histogram('{}/grad'.format(name.replace('.','/')),
                             grad.data,
                             module.global_step-1,
                             bins=bins)
    return hook

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
    def monitor_forward(module, input, output):
        # workaround for getting var names correctly when registering
        # backprop hooks on intermediate variables because for some
        # reason the hook function seems to be shared among the vars...
#         varnames_backward = []
#         varname_i = 0
#         def var_grad_hook(grad):
#             nonlocal varname_i # prevent name_i from becoming local in closure
#             name = varnames_backward[varname_i]
# #             print('Backward var:', module.global_step, name, grad.data.shape)
#             summary_writer.add_histogram('{}/grad'.format(name.replace('.','/')),
#                                          grad.data,
#                                          module.global_step-1,
#                                          bins=bins)
#             varname_i += 1

        # iterate over the state after the forward pass
        # registering backprop hooks on intermediate variables
        # (allowing us to not need to retain grads)
        # and the named parameters
        # as well as recording the forward prop activations
        # and the updates from the last iteration if possible
        for name, tensor in module.state_dict().items():
            if isinstance(tensor, ag.Variable): # it's an intermediate computation
                varnames_backward.insert(0, name)
                if track_grad:
                    tensor.register_hook(grad_hook(module, summary_writer, name))
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

    # workaround for getting var names correctly when registering
    # backprop hooks on parameters because for some
    # reason the hook function seems to be shared among them...
    param_names = [ name for name, _ in module.named_parameters()]
    # param_names_backward = param_names[::-1]
    # paramname_i = 0

#     def param_grad_hook(grad):
#         nonlocal paramname_i # prevent name_i from becoming local in closure
#         name = param_names_backward[paramname_i]
# #         print('Backward param:', module.global_step, name, grad.data.shape)
#         summary_writer.add_histogram('{}/grad'.format(name.replace('.','/')),
#                                      grad.data,
#                                      module.global_step-1,
#                                      bins=bins)
#         paramname_i = (paramname_i + 1) % len(param_names_backward)
    if track_grad:
        for name, param in zip(param_names, module.parameters()):
            param.register_hook(grad_hook(module, summary_writer, name))

    module.register_forward_hook(monitor_forward)

def fig2img(fig, dpi=None, closefig=True):
    if dpi is not None:
        fig.set_dpi(dpi)
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if closefig:
        plt.close(fig)
    return data

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
    run_name = config.get('run_name', '{}-{}'.format(start_time, host_name))
    run_comment = config.get('run_comment', None)
    if run_comment:
        run_name += '-{}'.format(run_comment)

    log_dir = config.get('log_dir', 'runs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    run_dir = os.path.join(log_dir, run_name)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    config['run_dir'] = run_dir

    writer = SummaryWriter(run_dir)

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

    rseed = config.get('random_seed', None)
    if rseed:
        random.seed(rseed)
        torch.manual_seed(rseed)

    writer.add_text(config['tag'], text, 0)
    return writer
