from tensorboardX import SummaryWriter

def set_monitor(module):
    """ Defines the monitor method on the module. """
    def monitor(name, tensor,
                track_data=True,
                track_grad=True):
        """
        Register the tensor under the name given (now a string)
        and track it based on the track_data and track_grad arguments.
        """
        module.monitored_vars[name] = {
            'tensor':tensor,
            'track_data':track_data,
            'track_grad':track_grad,
        }
    module.monitor = monitor

def set_monitoring(module):
    """ Defines the monitoring method on the module. """
    def monitoring(is_monitoring,
                   track_data=None,
                   track_grad=None,
                   track_update=None,
                   track_update_ratio=None):
        """
        Turn monitoring on or off. If any of the keyword arguments
        are not None, they will be overwritten.
        """
        module.is_monitoring = is_monitoring
        module.track_data = track_data if track_data is not None else module.track_data
        module.track_grad = track_grad if track_grad is not None else module.track_grad
        module.track_update = track_update if track_update is not None else module.track_update
        module.track_update_ratio = track_update_ratio if track_update_ratio is not None else module.track_update_ratio
    module.monitoring = monitoring

def grad_hook(module, name, writer, bins):
    """ Factory for grad_hook closures """
    def hook(grad):
        writer.add_histogram('{}/grad'.format(name.replace('.','/')),
                             grad.detach().cpu().numpy(),
                             module.global_step-1,
                             bins=bins)
    return hook

def remove_grad_hooks(module, input):
    """ Remove gradient hooks to all of the parameters and monitored vars """
    for hook in list(module.param_hooks.keys()):
        module.param_hooks[hook].remove()
        module.param_hooks.pop(hook)
    for hook in list(module.var_hooks.keys()):
        module.var_hooks[hook].remove()
        module.var_hooks.pop(hook)

def get_monitor_forward_and_backward(summary_writer, bins):
    """ Get the method for monitoring the forward values of the network """
    def monitor_forward_and_backward(module, input, output):
        """
        Iterate over the module parameters and monitor their forward values.
        Then iterate over all of the monitored_vars, monitor their forward values
        and set their grad_hooks
        """
        # Parameters
        if module.is_monitoring:
            param_names = [ name for name, _ in module.named_parameters()]
            for name, param in zip(param_names, module.parameters()):
                if module.track_grad and param.requires_grad:
                    hook = grad_hook(module, name, summary_writer, bins)
                    module.param_hooks[name] = param.register_hook(hook)
                if module.track_data:
                    summary_writer.add_histogram('{}/data'.format(name.replace('.','/')),
                                                 param.detach().cpu().numpy(),
                                                 module.global_step,
                                                 bins=bins)
                if name in module.last_state_dict:
                    if module.track_update:
                        update = param - module.last_state_dict[name]
                        summary_writer.add_histogram('{}/update-val'.format(name.replace('.','/')),
                                                     update.detach().cpu().numpy(),
                                                     module.global_step-1,
                                                     bins=bins)
                    if module.track_update and module.track_update_ratio:
                        update_ratio = update / (module.last_state_dict[name]+1e-15)

                        summary_writer.add_histogram('{}/update-ratio'.format(name.replace('.','/')),
                                                     update_ratio.detach().cpu().numpy(),
                                                     module.global_step-1,
                                                     bins=bins)
                if module.track_update:
                    module.last_state_dict[name] = param.clone()

            # Intermediate Vars
            for prefix, mod in module.named_modules():
                for tensor_name, entry in mod.monitored_vars.items():
                    name = '{}/{}'.format(prefix, tensor_name) if prefix else tensor_name
                    tensor = entry['tensor']
                    if entry['track_grad'] and tensor.requires_grad:
                        hook = grad_hook(module, name, summary_writer, bins)
                        module.var_hooks[name] = tensor.register_hook(hook)
                    if entry['track_data']:
                        summary_writer.add_histogram('{}/data'.format(name.replace('.','/')),
                                                     tensor.detach().cpu().numpy(),
                                                     module.global_step,
                                                     bins=bins)
            # Update step
        module.global_step += 1
    return monitor_forward_and_backward

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


    # The module will need additional information
    module.track_data = track_data
    module.track_grad = track_grad
    module.track_update = track_update
    module.track_update_ratio = track_update_ratio
    if not hasattr(module, 'global_step'):
        module.global_step = 0
    if not hasattr(module, 'is_monitoring'):
        module.is_monitoring = True
    if not hasattr(module, 'monitoring'):
        set_monitoring(module)
    if not hasattr(module, 'last_state_dict'):
        module.last_state_dict = dict()
    if not hasattr(module, 'var_hooks'):
        module.var_hooks = dict()
    if not hasattr(module, 'param_hooks'):
        module.param_hooks = dict()

    # All submodules need to have these
    for name, mod in module.named_modules():
        if not hasattr(mod, 'monitor'):
            set_monitor(mod)
        if not hasattr(mod, 'monitored_vars'):
            mod.monitored_vars = dict()

    module.monitoring(True)

    # remove previous grad hooks before handles go stale
    module.register_forward_pre_hook(remove_grad_hooks)

    # set forward hook that monitors forward activations and sets new grad hooks
    monitor_forward_and_backward = get_monitor_forward_and_backward(summary_writer, bins)
    module.register_forward_hook(monitor_forward_and_backward)
