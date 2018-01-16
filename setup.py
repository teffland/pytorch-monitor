from setuptools import setup
setup(
  name = 'pytorch-monitor',
  packages = ['pytorch_monitor'], # this must be the same as the name above
  version = '0.3',
  description = 'Monitor pytorch modules with minimal code.',
  author = 'Tom Effland',
  author_email = 'teffland@cs.columbia.edu',
  url = 'https://github.com/teffland/pytorch-monitor', # use the URL to the github repo
  download_url = 'https://github.com/teffland/pytorch-monitor/archive/0.3.tar.gz',
  keywords = ['pytorch', 'logging'], # arbitrary keywords
  classifiers = [],
  license='MIT',
  install_requires = [
    'sh',
    'tensorboardX'
  ]
)
