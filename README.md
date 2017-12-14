## When running an experiment

```python
from experiment_monitor import monitor_module, init_experiment

# an example config dict
config = dict(
    title="Dummy Code",
    description="Testing seq2seq with dummy but nontrivial data",
    log_dir='logs',
#     run_name='custom run name', # defaults to START_TIME-HOST_NAME
#     run_comment='custom run comment' # gets appended to run_name as RUN_NAME-RUN_COMMENT

    # hyperparams
    random_seed=42,
    learning_rate=.001,
    max_epochs=1000,
    batch_size=32,
    concrete_tau=10.,
    max_elbo_dev_steps=50,

    # model config
    word_vector_size=100,
)

# setup the experiment
augmented_config, writer = init_experiment(config)

# monitor the model
model = YourFancySchmancyModel()
monitor_module(model, writer)

# run your experiment...
...
```

## To monitor some intermediate computation in a model
```python
def forward(self, x):
  y = self.encode(x)

  # register intermediate computations to have them monitored
  self.register_buffer('y', y)

  z = self.decode(y)
  return z
```

the monitor is just an instance of [pytorch-tensorboard](http://tensorboard-pytorch.readthedocs.io/en/latest/tensorboard.html) so you can use any of those methods for anything extra you need.
