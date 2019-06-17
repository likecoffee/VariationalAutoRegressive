import pickle
import numpy as np
import collections

class BatchIter(object):
    def __init__(self, file_name,batch_size, dialogue_padded = 0, use_additional_column=True, additional_padded = 0):
        with open(file_name,"rb") as f:
            self.data_list = pickle.load(f)
        self.batch_size = batch_size
        self.data_size = len(self.data_list)
        self.num_batch = self.data_size // self.batch_size
        self.dialogue_padded = dialogue_padded
        if len(self.data_list[0]) > 2 and use_additional_column == True:
            self.additional_column = True
            self.additional_padded = additional_padded
        else:
            self.additional_column = False

    def __iter__(self):
        self.shuffled_index = np.arange(self.num_batch)
        np.random.shuffle(self.shuffled_index)
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.num_batch:
            raise StopIteration
        else:
            current_index = self.shuffled_index[self.index]
            start,end = current_index*self.batch_size,(current_index+1)*self.batch_size
            data_chunk = self.data_list[start:end]
            data_item_list = list(zip(*data_chunk))
            # Processing the dialogue text part
            source_list, target_list = data_item_list[0],data_item_list[1]
            source_len = np.array([len(source) for source in source_list])
            target_len = np.array([len(target) for target in target_list])
            source_max_len = np.max(source_len)
            target_max_len = np.max(target_len)
            sorted_index = np.argsort(source_len)[::-1]
            source_list = [source+[self.dialogue_padded]* (source_max_len-len(source)) for source in source_list]
            target_list = [target+[self.dialogue_padded]* (target_max_len-len(target)) for target in target_list]
            source_array = np.array(source_list,dtype=np.int).swapaxes(0,1)
            target_array = np.array(target_list,dtype=np.int).swapaxes(0,1)
            source_array = source_array[:,sorted_index]
            target_array = target_array[:,sorted_index]
            source_len = source_len[sorted_index]
            target_len = target_len[sorted_index]
            # Process the additional part
            if self.additional_column:
                additional_array_list = []
                for additional_data_item in data_item_list[2:]:
                    sample_item = additional_data_item[0]
                    if isinstance(sample_item, collections.Iterable) and not isinstance(sample_item,str):
                        len_array = np.array([len(item) for item in additional_data_item])
                        max_len = np.max(len_array)
                        padded_list = [item+[self.additional_padded]* (max_len-len(item)) for item in additional_data_item]
                        additional_array =np.array(padded_list).swapaxes(0,1)
                    elif isinstance(sample_item, int) or isinstance(sample_item, float):
                        additional_array = np.asarray(additional_data_item)
                    else:
                        raise NotImplementedError()

                    additional_array_list.append(additional_array)
            self.index += 1
            if self.additional_column:
                return self.index,source_array,target_array,source_len,target_len,additional_array_list
            else:
                return self.index,source_array,target_array,source_len,target_len
