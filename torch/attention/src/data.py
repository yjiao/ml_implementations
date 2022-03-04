"""
Utility functions for data importing, exporting, and tokenization.
"""
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
import os
from typing import List, Callable, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import pickle
import tokenizers
import logging
import math
from itertools import cycle
import random

TRAIN_PATH_EN = "data/train.en"
TRAIN_PATH_DE = "data/train.de"
TEST_PATH_EN = "data/test/newstest2014.en"
TEST_PATH_DE = "data/test/newstest2014.de"
TOKENIZER_PATH = "data/tokenizer-bbpe-joint.json"

PAD_TXT = "<pad>"
START_TXT = "<s>"
END_TXT = "<e>"


@contextmanager
def open_file_list(file_paths, *args, **kwargs):
    handles = [open(path, *args, **kwargs) for path in file_paths]
    yield handles
    [handle.close() for handle in handles]


def get_bbpe_tokenizer(
    tokenizer_path: str = TOKENIZER_PATH,
    data_file_list: List[str] = None,
    pad_txt=PAD_TXT,
    start_txt=START_TXT,
    end_txt=END_TXT,
):
    """Load or create byte-level byte pair encoding tokenizer and add special tokens."""
    if not data_file_list:
        data_file_list = [TRAIN_PATH_EN, TRAIN_PATH_DE]
    if not os.path.exists(tokenizer_path):
        logging.info(
            f"""Creating BBPE tokenizer:
saving to: {tokenizer_path}
reading from file list:
{file_paths}"""
        )
        bbpe = ByteLevelBPETokenizer()
        with open_file_list(file_paths, encoding="utf-8") as fhs:
            for fh in fhs:
                bbpe.train_from_iterator(fh)
    else:
        logging.info("loading BBPE tokenizer at %s", tokenizer_path)
        bbpe = tokenizers.Tokenizer.from_file(tokenizer_path)

    bbpe.add_special_tokens(
        [
            PAD_TXT,
            START_TXT,
            END_TXT,
        ]
    )
    return bbpe


@dataclass
class DatasetConfig:
    """Keep track of various configurations for datasets."""

    tokenizer: ByteLevelBPETokenizer
    max_seqlen: int
    start_txt: str = START_TXT
    end_txt: str = END_TXT
    pad_txt: str = PAD_TXT

    @property
    def start_id(self):
        return self.tokenizer.token_to_id(self.start_txt)

    @property
    def end_id(self):
        return self.tokenizer.token_to_id(self.end_txt)

    @property
    def pad_id(self):
        return self.tokenizer.token_to_id(self.pad_txt)


def trim_pad(lst: List[int], seqlen: int, pad_val: int) -> List[int]:
    """Trim and/or pad given list to length `seqlen`."""
    lst = lst[:seqlen]
    lst += [pad_val] * (seqlen - len(lst))
    return lst


def str_to_tok(
    input_str: str,
    config: DatasetConfig,
    add_start: bool = True,
):
    """Convert strings to tokens."""
    tokenizer = config.tokenizer

    toks = config.tokenizer.encode(input_str).ids

    if add_start:
        toks = [config.start_id] + toks
        seqlen = config.max_seqlen - 2
    else:
        seqlen = config.max_seqlen - 1

    # maybe trim, leave space for end token
    toks = toks[: config.max_seqlen - 1]
    toks.append(config.end_id)
    toks = trim_pad(toks, config.max_seqlen, config.pad_id)
    return toks


class WMT2014(torch.utils.data.IterableDataset):
    """WMT2014 dataset with multiprocessing.

    TODO: enable bucketing for input sequence lengths.
    """

    def __init__(
        self,
        *,
        start_line: int,
        end_line: int,
        process_inputs: Callable,
        process_target: Callable,
        dataset_config: DatasetConfig,
        train: bool = True,
        overwrite_en_path: Optional[str] = None,
        overwrite_de_path: Optional[str] = None,
    ):
        super().__init__()

        self.start = start_line
        self.end = end_line
        self.process_inputs = process_inputs
        self.process_target = process_target
        self.config = dataset_config
        self.train = train

        if train:
            self.en_path = overwrite_en_path or TRAIN_PATH_EN
            self.de_path = overwrite_de_path or TRAIN_PATH_DE
        else:
            self.en_path = overwrite_en_path or TEST_PATH_EN
            self.de_path = overwrite_de_path or TEST_PATH_DE

        self.en_linebreaks = self.get_line_breaks(self.en_path)
        self.de_linebreaks = self.get_line_breaks(self.de_path)

    def get_line_breaks(self, file_path):
        line_break_path = file_path + "_linebreaks.pickle"
        line_breaks = []
        if os.path.exists(line_break_path):
            logging.info(
                f"Loading existing line breaks for file {file_path} at {line_break_path}"
            )
            with open(line_break_path, "br") as lbp:
                line_breaks = pickle.load(lbp)
        else:
            logging.info(f"Regenerating line breaks for file {file_path}")
            offset = 0
            with open(self.paths[self.key_en], "r", encoding="utf-8") as fh:
                for line in fh:
                    line_breaks.append(offset)
                    offset += len(line)
            with open(line_break_path, "bw") as lbp:
                pickle.dump(line_breaks, lbp)
        return line_breaks

    def __len__(self):
        return self.end - self.start

    def get_fh_at_position(self, start, end):
        en_pos = self.en_linebreaks[start]
        de_pos = self.de_linebreaks[start]
        with open_file_list([self.en_path, self.de_path], "r", encoding="utf-8") as fhs:
            en_fh, de_fh = fhs
            en_fh.seek(en_pos)
            de_fh.seek(de_pos)

            for i in range(start, end):
                try:
                    en_line = en_fh.readline()
                    de_line = de_fh.readline()

                    en_toks = self.process_inputs(en_line, de_line, self.config)
                    de_toks = self.process_target(de_line, de_line, self.config)

                    yield [torch.tensor(en_toks), torch.tensor(de_toks)]
                except UnicodeDecodeError:
                    continue

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # main process, no multi-processing
            yield from self.get_fh_at_position(self.start, self.end)
        else:
            lines_per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            line_start = self.start + worker_id * lines_per_worker
            line_end = min(line_start + lines_per_worker, self.end)
            yield from self.get_fh_at_position(line_start, line_end)


class WMT2014Mixing(WMT2014):
    """WMT2014 dataset with mixing.

    The original data file exibits strange periodicity in the loss curves,
    indicating that several distinct populations of text exist in the file.
    Ex. some lines are not in English, some lines start with numbers, etc.
    """

    def __init__(self, *, num_pools: int, **kwargs):
        super().__init__(**kwargs)
        self.num_pools = num_pools
        self.make_pools()

    def make_pools(self):
        if self.num_pools > self.end - self.start:
            raise Exception(
                f"Number of pools ({self.num_pools}) is greater than the number of lines in the file ({end - start})."
            )
        self.pools = [None] * self.num_pools
        self.lines_per_pool = int(
            math.floor((self.end - self.start) / float(self.num_pools))
        )
        logging.info(
            f"IterShuffleMixin: Initializing {self.num_pools} pools with {self.lines_per_pool} lines per pool"
        )

    def __iter__(self):
        for i in range(self.num_pools):
            line_start = self.start + i * self.lines_per_pool
            line_end = min(line_start + self.lines_per_pool, self.end)
            self.pools[i] = self.get_fh_at_position(line_start, line_end)

        try:
            while True:
                random.shuffle(self.pools)
                for pool in self.pools:
                    yield next(pool)
        except StopIteration:
            pass
