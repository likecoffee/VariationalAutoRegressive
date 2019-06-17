import time
import pickle
import json
import numpy as np

class Configuration(object):
    def __init__(self, logging_file_name, learning_dict ,common_dict, encoder_dict, decoder_dict, interval_dict):
        self._logging_file_name = logging_file_name
        self._learning_dict = learning_dict
        self._interval_dict = interval_dict
        self._common_dict = common_dict
        self._encoder_dict = encoder_dict
        self._decoder_dict = decoder_dict

    @classmethod
    def load(cls, json_file_name):
        with open(json_file_name) as f:
            data_dict = json.load(f)
        return cls(data_dict['logging_file_name'],data_dict['learning'],data_dict['common'],data_dict['encoder'],data_dict['decoder'],data_dict['interval'])

    def save(self, json_file_name):
        data_dict = {"common":self._common_dict,"encoder":self._encoder_dict,"decoder":self._decoder_dict,
                     "learning":self._learning_dict,"interval":self._interval_dict}
        with open(json_file_name,"w") as wf:
            json.dump(data_dict, wf)

    def __repr__(self):
        learning_s= "Learning: \n" + "\n".join(["\t{0} : {1}".format(name,self._learning_dict[name]) for name in self._learning_dict.keys()])+"\n"
        common_s =  "Common:  \n" + "\n".join(["\t{0} : {1}".format(name,self._common_dict[name]) for name in self._common_dict.keys()])+"\n"
        encoder_s = "Encoder: \n" + "\n".join(["\t{0} : {1}".format(name,self._encoder_dict[name]) for name in self._encoder_dict.keys()])+"\n"
        decoder_s = "Decoder: \n" + "\n".join(["\t{0} : {1}".format(name,self._decoder_dict[name]) for name in self._decoder_dict.keys()])+"\n"

        return learning_s+common_s+encoder_s+decoder_s

    @property
    def common(self):
        return self._common_dict
    @property
    def encoder(self):
        encoder_config_dict = self._encoder_dict
        encoder_config_dict.update(self._common_dict)
        return encoder_config_dict
    @property
    def decoder(self):
        decoder_config_dict = self._decoder_dict
        decoder_config_dict.update(self._common_dict)
        return decoder_config_dict
    @property
    def learning(self):
        return self._learning_dict
    @property
    def interval(self):
        return self._interval_dict
    @property
    def logging_file_name(self):
        return self._logging_file_name
