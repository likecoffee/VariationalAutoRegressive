import logging
import time
import pickle
import json
import numpy as np

class Report(object):
    def __init__(self,report_names):
        self.report_names = report_names
        self.report_list = {name:[] for name in report_names}
        self.report_stamp = []
        self.report_start = 0
        self.report_time = time.time()

    def add_report_dict(self,report_dict):
        for name in report_dict.keys():
            if name in self.report_list.keys():
                self.report_list[name].append(report_dict[name])
        if self.report_stamp == []:
            self.report_stamp.append(1)
        else:
            self.report_stamp.append(self.report_stamp[-1]+1)

    def report_online(self,prefix_name=None):
        report_end = self.report_stamp[-1]
        mean_dict = {name:np.mean(self.report_list[name][self.report_start:report_end]) for name in self.report_names}
        logging_string_list = list()
        if prefix_name is None:
            logging_string_list.append("Batch : {0:8d}\t".format(report_end))
        else:
            logging_string_list.append(prefix_name+"\t")
        for name in mean_dict:
            logging_string_list.append("{0} : {1:6f} ".format(name,mean_dict[name]))
        logging_string_list.append("Elapsed: {0:4f}".format(time.time()-self.report_time))
        self.report_time = time.time()
        self.report_start = report_end
        logging.info("".join(logging_string_list))
    def save(self,file_name):
        with open(file_name,"wb") as wf:
            pickle.dump(self.report_list,wf)
