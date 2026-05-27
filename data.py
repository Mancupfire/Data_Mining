import os
import torch
from utils import *
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random

def load_split_data(config):
    def transform_token2id_seq(token_seqs, item2id):
        id_seqs = []
        for one_piece in token_seqs:
            item_token_seq = one_piece["inter_history"]
            item_id_seq = [item2id[token] for token in item_token_seq]
            target_id = item2id[one_piece["target_id"]]
            id_seqs.append(item_id_seq + [target_id])
        return id_seqs
            
    data_path = config["data_path"]
    dataset = config["dataset"]
    dataset_path = os.path.join(data_path, f"{dataset}/{dataset}")
    map_path = dataset_path + config["map_path"]
    
    train_inter = load_jsonl(dataset_path + ".train.jsonl")
    valid_inter = load_jsonl(dataset_path + ".valid.jsonl")
    test_inter = load_jsonl(dataset_path + ".test.jsonl")

    item2id = load_json(map_path) 
    
    train_seq = transform_token2id_seq(train_inter, item2id)
    valid_seq = transform_token2id_seq(valid_inter, item2id)
    test_seq = transform_token2id_seq(test_inter, item2id)

    n_items = len(item2id)
    return item2id, n_items, train_seq, valid_seq, test_seq
    
class SequentialSplitDataset(Dataset):
    def __init__(self, config, n_items, inter_seq, data_ratio=1, item_features_matrix=None):
        self.n_items = n_items
        self.config = config
        self.item_features_matrix = item_features_matrix
        self.max_length = config.get('max_length', 50)

        if data_ratio < 1:
            n_sample = int(len(inter_seq)*data_ratio)
            inter_seq = random.sample(inter_seq, n_sample)
            
        self.data = self.__map_inter__(inter_seq)

    def __map_inter__(self, inter_seq):
        data = []
        for seq in inter_seq:
            target = seq[-1]
            dict_data = {"id_seq": seq[:-1], "target": [target]}
            data.append(dict_data)
        return data
            
    def __getitem__(self, idx):
        data = self.data[idx]
        id_seq = data['id_seq']
        target = data['target']
        
        if len(id_seq) > self.max_length:
            id_seq = id_seq[-self.max_length:]

        item = {
            "id_seq": id_seq,
            "target": target
        }
        
        if self.item_features_matrix is not None:
            item["item_features"] = self.item_features_matrix[id_seq]
            
        return item

    def __len__(self):
        return len(self.data)
    
class Collator(object):
    def __init__(self, eos_token_id, pad_token_id, max_length):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.max_length = max_length
    
    def __pad_seq__(self, seq):
        if len(seq) > self.max_length:
            return seq[-self.max_length+1:]
        return seq
    
    def __call__(self, batch):
        id_seqs = [item["id_seq"] for item in batch]
        targets = [item["target"] for item in batch]
        
        input_ids = [torch.tensor(self.__pad_seq__(id_seq)) for id_seq in id_seqs]
        input_ids = pad_sequence(input_ids).transpose(0, 1)
        input_ids = input_ids.to(torch.long)

        attention_mask = (input_ids != self.pad_token_id).bool()
                              
        targets = torch.tensor(targets)
        targets = targets.to(torch.long).contiguous()
        
        output = dict(input_ids=input_ids,
                      attention_mask=attention_mask,
                      targets=targets)
                      
        if "item_features" in batch[0]:
            feature_dim = batch[0]["item_features"].shape[-1]
            batch_size = len(batch)
            pad_len = input_ids.shape[1] 
            
            padded_features = torch.zeros((batch_size, pad_len, feature_dim), dtype=torch.float32)
            for i, item in enumerate(batch):
                feat = item["item_features"]
                if len(feat) > self.max_length:
                    feat = feat[-self.max_length+1:]
                seq_len = len(feat)
                padded_features[i, :seq_len] = torch.tensor(feat, dtype=torch.float32)
                
            output["item_features"] = padded_features
            
        return output