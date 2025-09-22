#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : scene_graph.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/19/2018
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os
import collections

import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn

import functional_2d as functional_2d


DEBUG = bool(int(os.getenv('DEBUG_SCENE_GRAPH', 0)))

__all__ = ['SceneGraph2D', 'SceneGraphFeature']


class SceneGraphFeature(collections.namedtuple(
    '_SceneGraphFeature', ('scene_feature', 'object_feature', 'relation_feature')
)):
    pass


class SceneGraph2D(nn.Module):
    def __init__(self, feature_dim, output_dims, downsample_rate, pool_size=7, incl_relation=True, batch=False, norm=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dims = tuple(output_dims)
        self.downsample_rate = downsample_rate
        self.pool_size = pool_size
        self.incl_relation = incl_relation
        self.batch = batch
        self.norm = norm

        self.object_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)
        self.context_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)

        if self.incl_relation:
            self.relation_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)

        if not DEBUG:
            self.context_feature_extract = nn.Conv2d(feature_dim, feature_dim, 1)
            self.object_feature_fuse = nn.Conv2d(feature_dim * 2, output_dims[1], 1)
            self.object_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[1] * self.pool_size ** 2, output_dims[1]))

            if self.incl_relation:
                self.relation_feature_extract = nn.Conv2d(feature_dim, feature_dim // 2 * 3, 1)
                self.relation_feature_fuse = nn.Conv2d(feature_dim // 2 * 3 + output_dims[1] * 2, output_dims[2], 1)
                self.relation_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[2] * self.pool_size ** 2, output_dims[2]))
            self.reset_parameters()
        else:
            def gen_replicate(n):
                def rep(x):
                    return torch.cat(tuple(x for _ in range(n)), dim=1)
                return rep

            self.pool_size = 32
            self.object_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
            self.context_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
            self.relation_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
            self.context_feature_extract = gen_replicate(2)
            self.relation_feature_extract = gen_replicate(3)
            self.object_feature_fuse = jacnn.Identity()
            self.relation_feature_fuse = jacnn.Identity()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(
        self,
        input: torch.Tensor,
        objects: torch.Tensor,
        objects_length: torch.Tensor,
        object_collate=None
    ):
        object_features = input
        context_features = self.context_feature_extract(input)

        relation_features = None
        if self.incl_relation:
            relation_features = self.relation_feature_extract(input)

        objects_length = objects_length.detach().cpu()
        max_nr_objects = objects_length.max().item()

        if object_collate is None:
            if objects.dim() == 2:
                object_collate = 'concat'
            elif objects.dim() == 3:
                object_collate = 'pad'
            else:
                raise ValueError('Cannot infer object collates from objects.dim().')

        outputs = list()
        objects_index = 0  # for object_collate == 'concat' only
        for i in range(input.size(0)):
            if object_collate == 'concat':
                box = objects[objects_index:objects_index + objects_length[i].item()]
                objects_index += objects_length[i].item()
            else:
                box = objects[i, :objects_length[i].item()]

            if self.batch:
                if box.size(0) < max_nr_objects:
                    box = torch.cat((
                        box,
                        torch.zeros(max_nr_objects - box.size(0), box.size(1), dtype=box.dtype, device=box.device)
                    ), dim=0)

            with torch.no_grad():
                batch_ind = i + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device)

                # generate a "full-image" bounding box
                image_h, image_w = input.size(2) * self.downsample_rate, input.size(3) * self.downsample_rate
                image_box = torch.cat((
                    torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device),
                    torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device),
                    image_w + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device),
                    image_h + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device)
                ), dim=-1)

                box_context_imap = functional_2d.generate_intersection_map(box, image_box, self.pool_size)

                if self.incl_relation:
                    # meshgrid to obtain the subject and object bounding boxes
                    sub_id, obj_id = jactorch.meshgrid(torch.arange(box.size(0), dtype=torch.int64, device=box.device), dim=0)
                    sub_id, obj_id = sub_id.contiguous().view(-1), obj_id.contiguous().view(-1)
                    sub_box, obj_box = jactorch.meshgrid(box, dim=0)
                    sub_box = sub_box.contiguous().view(box.size(0) ** 2, 4)
                    obj_box = obj_box.contiguous().view(box.size(0) ** 2, 4)

                    # union box
                    union_box = functional_2d.generate_union_box(sub_box, obj_box)
                    rel_batch_ind = i + torch.zeros(union_box.size(0), 1, dtype=box.dtype, device=box.device)

                    # intersection maps
                    sub_union_imap = functional_2d.generate_intersection_map(sub_box, union_box, self.pool_size)
                    obj_union_imap = functional_2d.generate_intersection_map(obj_box, union_box, self.pool_size)

            this_context_features = self.context_roi_pool(context_features, torch.cat([batch_ind, image_box], dim=-1))
            x, y = this_context_features.chunk(2, dim=1)
            this_object_features = self.object_feature_fuse(torch.cat((
                self.object_roi_pool(object_features, torch.cat([batch_ind, box], dim=-1)),
                x, y * box_context_imap
            ), dim=1))

            if self.incl_relation:
                this_relation_features = self.relation_roi_pool(
                    relation_features,
                    torch.cat((rel_batch_ind, union_box), dim=-1)
                )
                x, y, z = this_relation_features.chunk(3, dim=1)
                this_relation_features = self.relation_feature_fuse(torch.cat((
                    this_object_features[sub_id], this_object_features[obj_id],
                    x, y * sub_union_imap, z * obj_union_imap
                ), dim=1))
            else:
                this_relation_features = None

            if not DEBUG:
                this_object_features = self._norm(self.object_feature_fc(this_object_features.view(box.size(0), -1)))
                if self.incl_relation:
                    this_relation_features = self._norm(self.relation_feature_fc(this_relation_features.view(box.size(0), box.size(0), -1)))

            outputs.append(SceneGraphFeature(None, this_object_features, this_relation_features))

        if self.batch:
            return self._batchify(outputs, objects_length, max_nr_objects)
        return outputs

    def _norm(self, x):
        if self.norm:
            return x / x.norm(2, dim=-1, keepdim=True)
        return x

    def _batchify(self, outputs, objects_length, max_nr_objects):
        output1 = torch.stack([x[1] for x in outputs], dim=0)
        mask1 = jactorch.length2mask(objects_length, max_nr_objects)
        if self.incl_relation:
            output2 = torch.stack([x[2] for x in outputs], dim=0)
            mask2_x, mask2_y = jactorch.meshgrid(mask1, dim=-1)
            mask2 = torch.max(mask2_x, mask2_y)
            return (None, (output1, mask1), (output2, mask2))
        else:
            return (None, (output1, mask1), None)