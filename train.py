import flair
import argparse
import json
import os
from flair.datasets import ColumnDataset
from flair.data import Corpus, FlairDataset, Sentence, Token
from flair.embeddings import (
  WordEmbeddings,
  StackedEmbeddings,
  FlairEmbeddings,
  TransformerWordEmbeddings
)
import subprocess
from flair.visual.training_curves import Plotter
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger

embeddings_map = {
  "glove": WordEmbeddings,
  "news-forward": FlairEmbeddings,
  "news-backward": FlairEmbeddings,
  "crawl": WordEmbeddings,
  "twitter": WordEmbeddings,
  'transformers': TransformerWordEmbeddings
}


def train(params):
  model_path = os.path.join(params["model_dir"], params["model_tag"])
  os.makedirs(model_path, exist_ok=True)
  # 1. get the corpus

  if len(params["filenames"]["train"]) > 1:
    train_file = os.path.join(model_path, "train.txt")
    p = subprocess.run("cat {} > {}".format(" ".join(params["filenames"]["train"]), train_file), shell=True,
                       stdout=subprocess.PIPE, universal_newlines=True)
    print(p.stdout)

  else:
    train_file = params["filenames"]["train"][0]

  train = ColumnDataset(path_to_column_file=train_file, tag_to_bioes="ner", column_name_map={0: "text", 1: "ner"})
  dev = ColumnDataset(path_to_column_file=params["filenames"]["dev"], tag_to_bioes="ner",
                      column_name_map={0: "text", 1: "ner"})
  test = ColumnDataset(path_to_column_file=params["filenames"]["test"], tag_to_bioes="ner",
                       column_name_map={0: "text", 1: "ner"})

  corpus: Corpus = Corpus(train, dev, test)
  print(corpus)

  # 2. what tag do we want to predict?
  tag_type = 'ner'

  # 3. make the tag dictionary from the corpus
  tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
  print(tag_dictionary.idx2item)

  embedding_types = []

  for emb in params["embeddings"]:
    if emb == "transformers":
      embedding_types.append(embeddings_map[emb](params["transformer_path"]))
    else:
      embedding_types.append(embeddings_map[emb](emb))

  embeddings = StackedEmbeddings(embeddings=embedding_types)
  print(embeddings)

  # initialize sequence tagger

  tagger = SequenceTagger(hidden_size=params["hidden_size"],
                          embeddings=embeddings,
                          tag_dictionary=tag_dictionary,
                          tag_type=tag_type,
                          use_rnn=params["use_rnn"],
                          rnn_layers=params["rnn_layers"],
                          use_crf=params["use_crf"],
                          )

  with open(os.path.join(model_path, 'config.json'), "w") as cfg:
    json.dump(params, cfg)
  # initialize trainer

  if os.path.exists(os.path.join(model_path, "checkpoint.pt")):
    trainer: ModelTrainer = ModelTrainer.load_checkpoint(os.path.join(model_path, "checkpoint.pt"),
                                                         corpus=corpus)
    max_epochs = params["max_epochs"] - trainer.epoch

  else:
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    max_epochs = params["max_epochs"]

  log_path = params["log_path"] if "log_path" in params else None
  if log_path is not None:
    log_path = os.path.join(params["log_path"], params["model_tag"])
    os.makedirs(log_path, exist_ok=True)

  trainer.train(log_path,
                model_path=model_path,
                learning_rate=params["learning_rate"],
                mini_batch_size=params["mini_batch_size"],
                max_epochs=max_epochs,
                save_final_model=params["save_model"],
                train_with_dev=params["train_with_dev"],
                anneal_factor=params["anneal_factor"],
                checkpoint=True)

  plotter = Plotter()
  plotter.plot_training_curves(os.path.join(log_path, "loss.tsv"))
  plotter.plot_weights(os.path.join(log_path, 'weights.txt'))


if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--config", default="config.json")
  args = arg_parser.parse_args()

  with open(args.config) as cfg:
    params = json.load(cfg)

  print(params)
  train(params)
