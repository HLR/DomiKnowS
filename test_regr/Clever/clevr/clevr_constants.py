#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : clevr_constants.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/29/2018
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Constants for the CLEVR dataset."""

from typing import Tuple, List

__all__ = ['g_attribute_concepts', 'g_relational_concepts', 'g_synonyms', 'load_clevr_concepts']


g_attribute_concepts = {
    'color': ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'],
    'material': ['rubber', 'metal'],
    'shape': ['cube', 'sphere', 'cylinder'],
    'size': ['small', 'large']
}

g_relational_concepts = {
    'spatial_relation': ['left', 'right', 'front', 'behind']
}

g_synonyms = {
    "thing": ["thing", "object"],
    "sphere": ["sphere", "ball", "spheres", "balls"],
    "cube": ["cube", "block", "cubes", "blocks"],
    "cylinder": ["cylinder", "cylinders"],
    "large": ["large", "big"],
    "small": ["small", "tiny"],
    "metal": ["metallic", "metal", "shiny"],
    "rubber": ["rubber", "matte"],
}


def load_clevr_concepts() -> Tuple[List[str], List[str], List[str]]:
    """Return the concepts for CLEVR dataset.

    Returns:
        Tuple[List[str], List[str], List[str]]: attribute_concepts, relational_concepts, multi_relational_concepts
    """

    attribute_concepts = []
    for k in g_attribute_concepts.keys():
        attribute_concepts.extend(g_attribute_concepts[k])
    for k in g_synonyms.keys():
        attribute_concepts.extend(g_synonyms[k])
    attribute_concepts = list(set(attribute_concepts))

    relational_concepts = g_relational_concepts['spatial_relation']
    multi_relational_concepts = g_relational_concepts['spatial_relation']

    return attribute_concepts, relational_concepts, multi_relational_concepts

