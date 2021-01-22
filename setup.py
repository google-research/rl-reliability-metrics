# coding=utf-8
# Copyright 2019 The Authors of RL Reliability Metrics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test and install rl-reliability-metrics."""
import datetime
import fnmatch
import io
import subprocess
import sys
import unittest

from setuptools import find_packages
from setuptools import setup
from setuptools.command.test import test as TestCommandBase
from setuptools.dist import Distribution


class StderrWrapper(io.IOBase):

  def write(self, *args, **kwargs):
    return sys.stderr.write(*args, **kwargs)

  def writeln(self, *args, **kwargs):
    if args or kwargs:
      sys.stderr.write(*args, **kwargs)
    sys.stderr.write('\n')


class TestLoader(unittest.TestLoader):

  def __init__(self, exclude_list):
    super(TestLoader, self).__init__()
    self._exclude_list = exclude_list

  def _match_path(self, path, full_path, pattern):
    if not fnmatch.fnmatch(path, pattern):
      return False
    module_name = full_path.replace('/', '.').rstrip('.py')
    if any(module_name.endswith(x) for x in self._exclude_list):
      return False
    return True


def load_test_list(filename):
  testcases = [
      x.rstrip() for x in open(filename, 'r').readlines()
      if x]
  # Remove comments and blanks after comments are removed.
  testcases = [x.partition('#')[0].strip() for x in testcases]
  return [x for x in testcases if x]


class Test(TestCommandBase):

  def run_tests(self):
    # Import absl inside run, where dependencies have been loaded already.
    from absl import app  # pylint: disable=g-import-not-at-top

    def main(_):
      run_separately = load_test_list('test_individually.txt')
      test_loader = TestLoader(exclude_list=run_separately)
      test_suite = test_loader.discover('rl_reliability_metrics',
                                        pattern='*_test.py')

      external_test_failures = []

      for test in run_separately:
        filename = 'rl_reliability_metrics.%s' % test
        try:
          # Using stderr=subprocess.STDOUT results in an exception in some
          # environments.
          subprocess.check_call(['python3', '-m', 'unittest', '-v', filename])
        except subprocess.CalledProcessError as e:
          external_test_failures.append(e)

      stderr = StderrWrapper()
      result = unittest.TextTestResult(stderr, descriptions=True, verbosity=2)
      test_suite.run(result)
      result.printErrors()

      for failure in external_test_failures:
        stderr.writeln('Individually run test failure:{}'.format(str(failure)))

      final_output = (
          'Tests run: {} grouped and {} external.  '.format(
              result.testsRun, len(run_separately)) +
          'Errors: {}  Failures: {}  External failures: {}.'.format(
              len(result.errors),
              len(result.failures),
              len(external_test_failures)))

      header = '=' * len(final_output)
      stderr.writeln(header)
      stderr.writeln(final_output)
      stderr.writeln(header)

      if result.wasSuccessful():
        return 0
      else:
        return 1

    # Run inside absl.app.run to ensure flags parsing is done.
    return app.run(main)


from rl_reliability_metrics.version import __dev_version__  # pylint: disable=g-import-not-at-top
from rl_reliability_metrics.version import __rel_version__  # pylint: disable=g-import-not-at-top

REQUIRED_PACKAGES = [
    'absl-py >= 0.6.1',
    'gin-config == 0.1.3',
    'numpy >= 1.13.3',
    'matplotlib >= 3.0.0',
    'six >= 1.10.0',
]


TEST_REQUIRED_PACKAGES = [
    'scipy == 1.4.1',
    'tensorflow == 2.1.0'
]


if '--release' in sys.argv:
  release = True
  sys.argv.remove('--release')
  version = __rel_version__
else:
  # Build a nightly package by default.
  release = False
  version = __dev_version__
  version += datetime.datetime.now().strftime('%Y%m%d')

if release:
  project_name = 'rl-reliability-metrics'
else:
  # Nightly releases use date-based versioning of the form
  # '0.0.1.dev20180305'
  project_name = 'rl-reliability-metrics-nightly'


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return False

#here = os.path.abspath(os.path.dirname(__file__))
#with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
#  long_description = f.read()

setup(
    name=project_name,
    version=version,
    description='RL-Reliability-Metrics: Reliability metrics for RL',
    # TODO: add long_description
    long_description=None,
    long_description_content_type='text/markdown',
    author='Google LLC',
    author_email='no-reply@google.com',
    url='http://github.com/google-research/rl-reliability-metrics',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    tests_require=TEST_REQUIRED_PACKAGES,
    extras_require={'tests': TEST_REQUIRED_PACKAGES},
    # Add in any packaged data.
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={
        'test': Test,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='agents reinforcement learning machine learning',
)
