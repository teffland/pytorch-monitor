from setuptools import setup
setup(
  name = 'pytorch-monitor',
  packages = ['pytorch-monitor'], # this must be the same as the name above
  version = '0.1',
  description = 'Monitor pytorch modules with minimal code.',
  author = 'Tom Effland',
  author_email = 'teffland@cs.columbia.edu',
  url = 'https://github.com/teffland/pytorch-monitor', # use the URL to the github repo
  download_url = 'https://github.com/teffland/pytorch-monitor/archive/0.1.tar.gz', # I'll explain this in a second
  keywords = ['pytorch', 'logging'], # arbitrary keywords
  classifiers = [],
  license='MIT',
  install_requires = [
    'tensorboardX'
  ]
)
