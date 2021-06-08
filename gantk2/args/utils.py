# Copyright 2021 Ibrahim Ayed, Emmanuel de Bézenac, Mickaël Chen, Jean-Yves Franceschi, Sylvain Lamprier, Patrick Gallinari

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import configargparse


def create_argparser():
    p = configargparse.ArgumentParser(
        prog='NTK GANs',
        description='NTK GANs.',
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter
    )
    return p


def create_default_parser_wrapper(create_args_fn):
    def create_parser(p=None):
        if p is None:
            p = create_argparser()
        return create_args_fn(p)
    return create_parser


def get_arg_group(p, title):
    groups = [g for g in p._action_groups if g.title == title]
    if len(groups) == 0:
        raise Warning('No groups with this title')
    elif len(groups) > 1:
        raise Warning('More than 1 group with the same title')
    return groups[0]
