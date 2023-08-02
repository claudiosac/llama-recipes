# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from .grammar_dataset import get_dataset as get_grammar_dataset
from .alpaca_dataset import InstructionDataset as get_alpaca_dataset
from .samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
from .qiansita_dataset import InstructionDataset as get_qiansita_dataset
from .qisumita_dataset import InstructionDataset as get_qisumita_dataset
from .qiinstructita_dataset import InstructionDataset as get_qiinstructita_dataset
from .qirphita_dataset import InstructionDataset as get_qirphita_dataset
from .qittlita_dataset import InstructionDataset as get_qittlita_dataset