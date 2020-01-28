from setuptools import setup
setup(
  name = 'pytorch-monitor',
  packages = ['pytorch_monitor'], # this must be the same as the name above
  version = '0.14',
  description = 'Monitor pytorch modules with minimal code.',
  author = 'Tom Effland',
  author_email = 'teffland@cs.columbia.edu',
  #url = 'https://github.com/teffland/pytorch-monitor', # use the URL to the github repo
  #download_url = 'https://github.com/teffland/pytorch-monitor/archive/0.14.tar.gz',
  keywords = ['pytorch', 'logging'], # arbitrary keywords
  classifiers = [],
  license='MIT',
  install_requires = [
    'sh',
    'tensorboardX',
    'tensorflow==1.15.2'
  ],
  python_requires='>=3'
)
