import torch
import copy
from torch.utils.data import TensorDataset
import tqdm

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids=None,input_len=None, rel_ix=None, sub_heads=None, sub_tails=None,obj_heads=None,obj_tails=None,sub_head=None,sub_tail=None,
                 start_mask=None, end_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_len = input_len
        self.rel_ix = rel_ix
        self.sub_heads = sub_heads
        self.sub_tails = sub_tails
        self.obj_heads = obj_heads
        self.obj_tails = obj_tails
        self.sub_head = sub_head
        self.sub_tail = sub_tail
        self.start_mask = start_mask
        self.end_mask = end_mask
def convert_examples_to_features(args, examples=None,tokenizer=None, data_type=None):
    if data_type!='test':
        features = tran_examples_to_features1(args, examples=examples,tokenizer=tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        token_type_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        sub_heads = torch.tensor([f.sub_heads for f in features], dtype=torch.long)
        sub_tails = torch.tensor([f.sub_tails for f in features], dtype=torch.long)
        sub_head = torch.tensor([f.sub_head for f in features], dtype=torch.long)
        sub_tail = torch.tensor([f.sub_tail for f in features], dtype=torch.long)
        obj_heads = torch.tensor([f.obj_heads for f in features], dtype=torch.long)
        obj_tails = torch.tensor([f.obj_tails for f in features], dtype=torch.long)
        rel_ix = torch.tensor([f.rel_ix for f in features], dtype=torch.long)
        dataset = TensorDataset(input_ids, attention_mask, token_type_ids, input_lens,rel_ix, sub_heads,
                                sub_tails,obj_heads,obj_tails,sub_head,sub_tail)
    else:
        features = tran_examples_to_features2(args, examples=examples,tokenizer=tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        start_mask = torch.tensor([f.start_mask for f in features], dtype=torch.long)
        end_mask = torch.tensor([f.end_mask for f in features], dtype=torch.long)
        dataset = TensorDataset(input_ids,attention_mask,input_lens,start_mask,end_mask)
    return dataset
def tran_examples_to_features2(args, examples=None,tokenizer=None):
    features = []
    pad_token = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    for e_ix, example in tqdm.tqdm(enumerate(examples)):
        text = example.text
        tokens = []
        start_mask = []
        end_mask = []
        for i,word in enumerate(text.split()):
            tmp_words = tokenizer.tokenize(word)
            if not tmp_words:
                tmp_words = ['[UNK]']
            tokens.extend(tmp_words)
            if len(tmp_words)==1:
                start_mask.append(1)
                end_mask.append(1)
            else:
                start_mask.append(1)
                start_mask.extend([0]*(len(tmp_words)-1))
                end_mask.extend([0]*(len(tmp_words)-1))
                end_mask.append(1)
        if len(tokens) > 510:
            tokens = tokens[:510]
            start_mask = start_mask[:510]
            end_mask = end_mask[:510]
        tokens += ['[SEP]']
        start_mask.append(0)
        end_mask.append(0)
        tokens = ['[CLS]']+tokens
        input_len = len(tokens)
        start_mask = [0]+start_mask
        end_mask = [0]+end_mask
        attention_mask = [1]*len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding_length = 512 - len(input_ids)
        input_ids += [pad_token] * padding_length
        attention_mask += [0] * padding_length
        start_mask += [0]*padding_length
        end_mask += [0]*padding_length
        features.append(InputFeatures(input_ids,attention_mask,input_len=input_len,start_mask=start_mask,end_mask=end_mask))
    return features


def tran_examples_to_features1(args, examples=None,tokenizer=None):
    pad_token = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    features = []
    for e_ix,example in tqdm.tqdm(enumerate(examples)):
        text = example.text
        subs = example.subs
        objs = example.objs
        rela = example.rela
        sub_headss = example.sub_heads
        sub_tailss = example.sub_tails
        tokens = []
        sub_tails = []
        sub_heads = []
        obj_head = []
        obj_tail = []
        sub_head = []
        sub_tail = []
        # print("subs:{}".format(subs))
        # print("objs:{}".format(objs))
        # print("sub_heasss:{}".format(sub_headss))
        # print("sub_tailss:{}".format(sub_tailss))
        # print("rela:{}".format(rela[0]))
        for i, word in enumerate(text.split()):
            tmp_words = tokenizer.tokenize(word)
            if not tmp_words:
                tmp_words = ['[UNK]']
            tokens.extend(tmp_words)
            # 给sub_head赋值，每次只有一个subject
            if i==subs[0]:
                sub_head.append(1)
                sub_head.extend([0] * (len(tmp_words) - 1))
            else:
                sub_head.extend([0] * len(tmp_words))
            # 给sub_tail赋值，每次只有一个subject
            if i==subs[1]:
                sub_tail.extend([0] * (len(tmp_words) - 1))
                sub_tail.append(1)
            else:
                sub_tail.extend([0] * len(tmp_words))
            # 给sub_heads赋值，每次可以有多个
            if i in sub_headss:
                sub_heads.append(1)
                sub_heads.extend([0]*(len(tmp_words)-1))
            else:
                sub_heads.extend([0]*len(tmp_words))
            # 给sub_tails赋值，每次可以有多个
            if i in sub_tailss:
                sub_tails.extend([0]*(len(tmp_words)-1))
                sub_tails.append(1)
            else:
                sub_tails.extend([0] * len(tmp_words))
            # 给obj_head赋值，每次只有一个object
            if i == objs[0]:
                obj_head.append(1)
                obj_head.extend([0]*(len(tmp_words)-1))
            else:
                obj_head.extend([0]*len(tmp_words))
            # 给obj_tail赋值，每次只有一个object
            if i == objs[1]:
                obj_tail.extend([0]*(len(tmp_words)-1))
                obj_tail.append(1)
            else:
                obj_tail.extend([0] * len(tmp_words))
        re_ix = args.rel2id[rela[0]]
        obj_heads = []
        obj_tails = []
        rel_num = args.num_rel
        for j in range(rel_num):
            if j==re_ix:
                obj_heads.append(copy.deepcopy(obj_head))
                obj_tails.append(copy.deepcopy(obj_tail))
            else:
                obj_heads.append([0]*len(obj_head))
                obj_tails.append([0]*len(obj_tail))
        if len(tokens) > 510:
            tokens = tokens[:510]
            sub_tails = sub_tails[:510]
            sub_heads = sub_heads[:510]
            sub_head = sub_head[:510]
            sub_tail = sub_tail[:510]
            obj_heads = [x[:510] for x in obj_heads]
            obj_tails = [x[:510] for x in obj_tails]
        tokens += ['[SEP]']
        sub_heads.append(0)
        sub_tails.append(0)
        sub_head.append(0)
        sub_tail.append(0)
        obj_heads = [x+[0] for x in obj_heads]
        obj_tails = [x+[0] for x in obj_tails]
        tokens = ['[CLS]']+tokens
        sub_tails = [0]+sub_tails
        sub_heads = [0]+sub_heads
        sub_head = [0]+sub_head
        sub_tail = [0]+sub_tail
        obj_heads = [[0]+x for x in obj_heads]
        obj_tails = [[0]+x for x in obj_tails]
        attention_mask = [1]*len(tokens)
        segment_ids = [0]*len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_len = len(input_ids)

        padding_length = 512 - len(input_ids)
        input_ids += [pad_token] * padding_length
        attention_mask += [0] * padding_length
        segment_ids += [0] * padding_length
        sub_tails += [0] * padding_length
        sub_heads += [0] * padding_length
        sub_tail += [0] * padding_length
        sub_head += [0] * padding_length
        obj_heads = [x+[0] * padding_length for x in obj_heads]
        obj_tails = [x+[0] * padding_length for x in obj_tails]
        # if e_ix==0:
        #   break
        features.append(InputFeatures(input_ids,attention_mask,segment_ids,input_len,re_ix,sub_heads,sub_tails,obj_heads,obj_tails,sub_head,sub_tail))
    return features









