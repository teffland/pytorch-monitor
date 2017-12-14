## Pytorch Monitor

So you've coded up some fancy network to do a some cool task.
You set up your training and evaluation scripts and click go, and come back 12 hours later.

Your losses and metrics aren't going up or down the way you expected, and you don't know why.
Incorrect gradients? Bad weight initializations? A bug in the forward pass of your network?

Without proper monitoring, it's hard to be sure.
But setting up all of that infrastructure is often way too much work.

Pytorch Monitor aims to do this with only a few lines of code.

The package does the following things:

* We use [tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) to render all the plots, histograms, etc, you want to log.
* We use the [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) library to perform the logging from pytorch.
* You initialize experiments with a config dict using `init_experiment` and we write them to tensorboard as formatted text, this way you can always check on any experiment details from tensorboard
  * We also will automatically commit your repo and log the commit hash so you can always recover the exact state of your code if needed.
* You wrap your network with `monitor_module` and we introspect it, automatically tracking the forward and backward passes of your parameters, as well as any intermediate computations you mark as important with `register_buffer`.

The end result is you can run your experiments as usual and get pretty plots and detailed logging, all with a couple extra lines of code.

## Installation

To install, you'll need to install:

1. [pytorch](http://pytorch.org/)
2. tensorflow: `pip install tensorflow`
3. pytorch-monitor: `pip install pytorch-monitor`

## Examples:

### When running an experiment

```python
from pytorch_monitor import monitor_module, init_experiment

# an example config dict
config = dict(
    title="An Experiment",
    description="Testing out a NN",
    log_dir='logs',
#     run_name='custom run name', # defaults to START_TIME-HOST_NAME
#     run_comment='custom run comment' # gets appended to run_name as RUN_NAME-RUN_COMMENT

    # hyperparams
    random_seed=42,
    learning_rate=.001,
    max_epochs=1000,
    batch_size=32,

    # model config
    n_hidden=100,
)

# setup the experiment
writer = init_experiment(config)

# monitor the model
model = YourFancySchmancyModel(config['n_hidden'])
monitor_module(model, writer)

# run your experiment...
```

### To monitor some intermediate computation in a model
```python
def forward(self, x):
  y = self.encode(x)

  # register intermediate computations to have them monitored
  self.register_buffer('y', y)

  z = self.decode(y)
  return z
```
