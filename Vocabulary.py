import numpy as np
import pickle

def load_vocabulary(file_name):
    with open(file_name,"rb") as f:
        return pickle.load(f)
def save_vocabulary(vocab,file_name):
    with open(file_name,"wb") as f:
        pickle.dump(vocab,f,protocol=2)

def statistic_truncated(vocab):
    truncated_num = np.sum([vocab.word_count[word] for word in list(vocab.truncated_word_id.keys())])
    total_num = np.sum(list(vocab.word_count.values()))
    unk_num = total_num-truncated_num
    top_vocab_list = vocab.sorted_word_count_tuple_list[:10]
    ratio_tuple_list = [(word,float(unk_num)/num) for word,num in top_vocab_list]
    print((float(truncated_num)/float(total_num)*100))
    print(ratio_tuple_list)

def load_embedding(word_id_dict,embedding_file_name="/home/dujiachen/local/embeddings/glove.840B.300d.txt",embedding_size=300):
    embedding_length = len(word_id_dict)+1
    embedding_matrix = np.random.uniform(-1e-2,1e-2,size=(embedding_length,embedding_size))
    embedding_matrix[0] = 0
    hit = 0
    with open(embedding_file_name,"r") as f:
        for line in f:
            splited_line = line.strip().split(" ")
            word,embeddings = splited_line[0],splited_line[1:]
            if word in word_id_dict:
                word_index = word_id_dict[word]
                embedding_array = np.fromstring("\n".join(embeddings),dtype=np.float32,sep="\n")
                embedding_matrix[word_index] = embedding_array
                hit += 1
    hit_rate = float(hit)/embedding_length
    print(("The hit rate is {}".format(hit_rate)))
    return embedding_matrix
        
class Vocabulary(object):
    def __init__(self):
        super(Vocabulary, self).__init__()
        self.word_id = {}
        self.word_count = {}
        self.word_id_current = 1
        self.unknown_id = 0
        self.unknown_sym = "UNK"
    
    def __str__(self):
        overall_s = "Vocabulary contains {}".format(self.word_id_current)
        if hasattr(self,"truncated_word_id"):
            truncated_s = "Truncated (cannot append word list) dictionary contains {}".format(len(self.truncated_word_id))
        else:
            truncated_s = ""
        return "\n".join((overall_s,truncated_s))
    
    def load_embedding(self,emb_path="/home/dujiachen/data/embeddings/glove_dict.pkl"):
        with open(emb_path,"rb") as f:
            embedding_dict = pickle.load(f)
        
    
    @property
    def truncated_id_word(self):
        if hasattr(self,"_truncated_id_word"):
            return self._truncated_id_word
        elif hasattr(self,"truncated_word_id"):
            self._truncated_id_word = {self.truncated_word_id[key]:key for key in list(self.truncated_word_id.keys())}
            return self._truncated_id_word
        else:
            raise Exception("hasn't construct truncated dictionary")
    @property
    def id_word(self):
        if hasattr(self,"_id_word"):
            return self._id_word
        elif hasattr(self,"word_id"):
            self._id_word = {value:self.word_id[key] for key in list(self.word_id.keys())}
            return self._id_word
        else:
            raise Exception("word has'nt been added")
        
    def convert_id_list(self,id_list,mode="truncated"):
        if mode == "truncated":
            lookup_dict = self.truncated_id_word
        elif mode == "full":
            lookup_dict = self.id_word
        else:
            raise ValueError("Wrong Mode")
        assert lookup_dict != None
        word_list = []
        for id in id_list:
            if id in lookup_dict:
                word_list.append(lookup_dict[id])
            else:
                word_list.append(self.unknown_sym)
        return word_list

                
    def add_word_list(self, word_list):
        if hasattr(self,"truncated_word_id"):
            raise Exception("Can't add word list")
        for word in word_list:
            if word not in self.word_id and word != self.unknown_sym:
                self.word_id[word] = self.word_id_current
                self.word_id_current += 1
                self.word_count[word] = 0
            self.word_count[word] += 1

    def convert_word_list(self, word_list,mode):
        return_list = []
        if mode=="full":
            for word in word_list:
                if word in self.word_id:
                    return_list.append(self.word_id[word])
                else:
                    return_list.append(unknown_id)
            return return_list
        elif mode=="truncated":
            if hasattr(self,"truncated_word_id"):
                for word in word_list:
                    if word in self.truncated_word_id:
                        return_list.append(self.truncated_word_id[word])
                    else:
                        return_list.append(self.unknown_id)
                return return_list
            else:
                raise Exception("truncated id dictionary doen't exist")
        else:
            raise Exception("mode must be full or truncated")
                
    
    def truncate_dictionary(self,num):
        self.truncated_length = num
        if not hasattr(self,"sorted_word_count"):
            word_count_tuple_list = [(key,self.word_count[key]) for key in list(self.word_count.keys())]
            self.sorted_word_count_tuple_list = sorted(word_count_tuple_list,key = lambda x: x[1],reverse=True)
        if type(num) is int:
            self.truncated_word_id = {self.sorted_word_count_tuple_list[index][0]:index+1 for index in range(num-1)}
        else:
            raise Exception("NOW ONLY SUPPORT INTEGER")
