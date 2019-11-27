# coding=utf-8
# Copyright 2019 The rl-reliability-metrics Authors.
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

"""OSS replacements for gfile io methods."""
import glob
import os


def makedirs(path):
  """Makes directory if it does not already exist."""
  if not os.path.exists(path):
    return os.makedirs(path)


def listdir(path):
  """Returns a list of directories."""
  return os.listdir(path)


def paths_glob(pattern):
  """Returns a list of directories."""
  return glob.glob(pattern)
