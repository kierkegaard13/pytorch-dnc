#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from torch.nn.init import orthogonal, xavier_uniform

from .util import *
from .sparse_memory import SparseMemory

from .dnc import DNC


class FFSAM(DNC):

    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            batch_size=32,
            num_layers=1,
            num_hidden_layers=2,
            bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=False,
            nr_cells=5000,
            sparse_reads=4,
            read_heads=4,
            cell_size=10,
            nonlinearity='tanh',
            gpu_id=-1,
            independent_linears=False,
            share_memory=True,
            debug=False,
            clip=20
    ):

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            nr_cells=nr_cells,
            read_heads=read_heads,
            cell_size=cell_size,
            nonlinearity=nonlinearity,
            gpu_id=gpu_id,
            independent_linears=independent_linears,
            share_memory=share_memory,
            define_layers=False,
            debug=debug,
            clip=clip
        )
        self.batch_size = batch_size
        self.sparse_reads = sparse_reads
        self.output_size = output_size

        # override SDNC memories with SAM
        self.memories = []
        self.model = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.nn_output_size, self.output_size),
            nn.LogSoftmax(dim=1)
        )
        if self.gpu_id != -1:
            self.model.cuda()
        self.ptm = [False, False, True, False, False, False]
        self.write_enabled = True

        for layer in range(self.num_layers):
          # memories for each layer
            if not self.share_memory:
                self.memories.append(
                    SparseMemory(
                        input_size=self.hidden_size,
                        mem_size=self.nr_cells,
                        cell_size=self.w,
                        sparse_reads=self.sparse_reads,
                        read_heads=self.read_heads,
                        gpu_id=self.gpu_id,
                        mem_gpu_id=self.gpu_id,
                        independent_linears=self.independent_linears
                    )
                )
                setattr(self, 'rnn_layer_memory_' + str(layer), self.memories[layer])

        # only one memory shared by all layers
        if self.share_memory:
            self.memories.append(
                SparseMemory(
                    input_size=self.hidden_size,
                    mem_size=self.nr_cells,
                    cell_size=self.w,
                    sparse_reads=self.sparse_reads,
                    read_heads=self.read_heads,
                    gpu_id=self.gpu_id,
                    mem_gpu_id=self.gpu_id,
                    independent_linears=self.independent_linears
                )
            )
            setattr(self, 'rnn_layer_memory_shared', self.memories[0])

    def train(self, mode=True):
        super().train(mode)
        self.write_enabled = True

    def eval(self):
        super().eval()
        self.write_enabled = False

    def _debug(self, mhx, debug_obj):
        if not debug_obj:
            debug_obj = {
                'memory': [],
                'visible_memory': [],
                'read_weights': [],
                'write_weights': [],
                'read_vectors': [],
                'least_used_mem': [],
                'usage': [],
                'read_positions': []
            }

        debug_obj['memory'].append(mhx['memory'][0].data.cpu().numpy())
        debug_obj['visible_memory'].append(mhx['visible_memory'][0].data.cpu().numpy())
        debug_obj['read_weights'].append(mhx['read_weights'][0].unsqueeze(0).data.cpu().numpy())
        debug_obj['write_weights'].append(mhx['write_weights'][0].unsqueeze(0).data.cpu().numpy())
        debug_obj['read_vectors'].append(mhx['read_vectors'][0].data.cpu().numpy())
        debug_obj['least_used_mem'].append(mhx['least_used_mem'][0].unsqueeze(0).data.cpu().numpy())
        debug_obj['usage'].append(mhx['usage'][0].unsqueeze(0).data.cpu().numpy())
        debug_obj['read_positions'].append(mhx['read_positions'][0].unsqueeze(0).data.cpu().numpy())

        return debug_obj

    def _init_hidden(self, mhx, reset_experience):
        # memory states
        if mhx is None:
            if self.share_memory:
                mhx = self.memories[0].reset(self.batch_size, erase=reset_experience)
            else:
                mhx = [m.reset(self.batch_size, erase=reset_experience) for m in self.memories]
        elif self.write_enabled:
            if self.share_memory:
                mhx = self.memories[0].reset(self.batch_size, mhx, erase=reset_experience)
            else:
                mhx = [m.reset(self.batch_size, h, erase=reset_experience) for m, h in zip(self.memories, mhx)]

        return mhx

    def _layer_forward(self, input, layer, mhx=None, pass_through_memory=True):
        # pass through the controller layer
        input = self.model[layer](input)

        # clip the controller output
        if self.clip != 0:
            output = T.clamp(input, -self.clip, self.clip)
        else:
            output = input

        # the interface vector
        ξ = output

        # pass through memory
        if pass_through_memory:
            if self.share_memory:
                read_vecs, mhx = self.memories[0](ξ, mhx, self.write_enabled)
            else:
                read_vecs, mhx = self.memories[layer](ξ, mhx, self.write_enabled)
            # the read vectors
            read_vectors = read_vecs.view(-1, self.w * self.r)
        else:
            read_vectors = None

        return output, (mhx, read_vectors)

    def forward(self, input, mhx, reset_experience=False, pass_through_memory=True):
        mem_hidden = self._init_hidden(mhx, reset_experience)
        inputs = input

        # pass thorugh layers
        for layer in range(len(self.model)):
            # this layer's hidden states
            mhx = mem_hidden if self.share_memory else mem_hidden[layer]
            # pass through controller
            outs, (mhx, read_vectors) = self._layer_forward(inputs, layer, mhx, self.ptm[layer])

            # store the memory back (per layer or shared)
            if self.share_memory:
                mem_hidden = mhx
            else:
                mem_hidden[layer] = mhx

            if read_vectors is not None:
                # the controller output + read vectors go into next layer
                outs = T.cat([outs, read_vectors], 1)
            inputs = outs

        # pass through final output layer
        outputs = inputs

        return outputs, mem_hidden
