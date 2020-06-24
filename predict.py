from flair.data import Sentence
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

    columns = {0: 'text', 1: 'ner'}
    args = argparser.parse_args()
    sentences_test = NLPTaskDataFetcher.read_column_data(args.ts, columns)

    for sentence in sentences_test:
        sentence = sentence
        sentence.convert_tag_scheme(tag_type="ner", target_scheme='iobes')

    tagger = SequenceTagger.load(args.model)

    res = tagger.evaluate(sentences_test, args.out)

    print(res[0].detailed_results)
