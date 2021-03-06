import logging
import inspect
from gpu_mem_track import MemTracker
import pickle
import os
from os.path import join, exists
from ipdb import launch_ipdb_on_exception
#from torchsummary import summary
import train
import configuration
import report
import Vocabulary
import seq2seqVAD
#import seq2seqVAD
import dataIter


def main():
    config = configuration.Configuration.load("config.json")
    if config.logging_file_name is not None:
        logging.basicConfig(level=logging.INFO,format="%(message)s",filename=config.logging_file_name,filemode="w")
    else:
        logging.basicConfig(level=logging.INFO,format="%(message)s")
    vocab = Vocabulary.load_vocabulary("./data/weibo_vocab_word.pkl")
    config.common["num_word"] = vocab.truncated_length
    frame = inspect.currentframe()
    gpu_tracker = MemTracker(frame)
    gpu_tracker.track()
    model = seq2seqVAD.Seq2SeqVAD(config)
    gpu_tracker.track()
    with open("./data/weibo_embedding_word.pkl", "rb") as f:
        embedding = pickle.load(f)
    model.init_weight(embedding)
    logging.info(repr(config))
    logging.info(repr(model))
    batch_size = config.learning["batch_size"]
    train_batch_generator = dataIter.BatchIter(
        "data/weibo_train_word.pkl", batch_size)
    valid_batch_generator = dataIter.BatchIter(
        "data/weibo_test_word.pkl", batch_size)
    test_batch_generator = dataIter.BatchIter(
        "data/weibo_test_word.pkl", 10)

    train_report_names = ["ce", "bwd_rnn_kld", "aux_bow", "bwd_ce",
                          "loss", "KLD_weight", "aux_weight", "bwd_ce_weight"]
    train_report = report.Report(train_report_names)
    valid_report_names = ["ce"]
    generation_report_names = ["emb_avg", "emb_ext",
                               "emb_gre", "dist-1", "dist-2", "novel"]
    generation_report = report.Report(generation_report_names)
    valid_report = report.Report(valid_report_names)
    parent_name = config.learning["parent_name"]
    if not exists(parent_name):
        os.mkdir(parent_name)
        os.mkdir(join(parent_name, "report"))
        os.mkdir(join(parent_name, "stochastic_array"))
        os.mkdir(join(parent_name, "saved_models"))
        os.mkdir(join(parent_name, "generated_dialogues"))
    
    train.train_model(model, config, train_batch_generator=train_batch_generator,
                      valid_batch_generator=valid_batch_generator, test_batch_generator=test_batch_generator,
                      vocab=vocab, embedding=embedding, report_train=train_report, report_valid=valid_report,
                      report_generation=generation_report, parent_file_name=parent_name)

if __name__ == "__main__":
    with launch_ipdb_on_exception():
        main()
