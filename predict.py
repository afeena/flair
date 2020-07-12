from flair.data import Sentence
from flair.datasets import ColumnDataset
from flair.models import SequenceTagger
from flair.data_fetcher import NLPTaskDataFetcher
import torch
from pathlib import Path
import argparse
import os
import re

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model")
    argparser.add_argument("--ts")
    argparser.add_argument("--out")

    args = argparser.parse_args()
    sentences_test = ColumnDataset(path_to_column_file=args.ts, tag_to_bioes="ner", column_name_map={0: "text", 1: "ner"})
   
    tagger = SequenceTagger.load(args.model)

    res = tagger.evaluate(sentences_test, args.out)

    print(res[0].detailed_results)
