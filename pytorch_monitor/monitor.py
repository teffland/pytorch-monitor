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
            if param.requires_grad:
                hook = grad_hook(module, name, summary_writer, bins)
                module.param_hooks[name] = param.register_hook(hook)
            
    # monitor forward grads
    module.register_forward_pre_hook(remove_old_var_hooks)
    module.register_forward_hook(monitor_forward_and_vars)