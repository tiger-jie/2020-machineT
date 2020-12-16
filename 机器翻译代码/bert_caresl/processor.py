import os
import pickle
import torch

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    input_ids, attention_mask, token_type_ids,  all_lens,re_ix, sub_heads,sub_tails, \
    obj_heads, obj_tails, sub_head, sub_tail = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]
    token_type_ids = token_type_ids[:, :max_len]
    sub_heads = sub_heads[:, :max_len]
    sub_tails = sub_tails[:, :max_len]
    sub_head = sub_head[:, :max_len]
    sub_tail = sub_tail[:, :max_len]
    obj_heads = obj_heads[:, :, :max_len]
    obj_tails = obj_tails[:, :, :max_len]
    return input_ids, attention_mask, token_type_ids,sub_heads,sub_tails,obj_heads,obj_tails,sub_head,sub_tail,re_ix
def collate_fn1(batch):
    input_ids, attention_mask,input_lens,start_mask,end_mask = map(torch.stack, zip(*batch))
    max_len = max(input_lens).item()
    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]
    start_mask = start_mask[:, :max_len]
    end_mask = end_mask[:, :max_len]
    return input_ids,attention_mask,start_mask,end_mask
class InputExample(object):
    def __init__(self, text, subs=None, objs=None, rela=None, sub_heads=None, sub_tails=None):
        self.text = text
        self.subs = subs
        self.objs = objs
        self.rela = rela
        self.sub_heads  =sub_heads
        self.sub_tails = sub_tails

class REProcessor(object):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read(os.path.join(data_dir, "train.pk")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read(os.path.join(data_dir, "dev.pk")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read(os.path.join(data_dir, "test.pk")), "test")

    def get_rels(self):
        return ['PositivelyRegulates','causes','induce','provide','ChemicalSynthess']

    def _create_examples(self, data, set_type):
        examples = []
        if set_type!='test':
            texts_ = data['text']
            sub_headss_ = data['start_h']
            sub_tailss_ = data['end_h']
            obj_headss_ = data['start_t']
            obj_tailss_ = data['end_t']
            relass_ = data['relas']
            text_ = []
            subs = []
            objs = []
            relas_ = []
            sub_heads = []
            sub_tails = []
            j = 0
            while j<len(texts_):
                if len(sub_headss_[j])==1:
                    sub_ = []
                    obj_ = []
                    rea_ = []
                    text_.append(texts_[j])
                    sub_.append(sub_headss_[j][0])
                    sub_.append(sub_tailss_[j][0])
                    obj_.append(obj_headss_[j][0])
                    obj_.append(obj_tailss_[j][0])
                    rea_.append(relass_[j][0])
                    subs.append(sub_)
                    objs.append(obj_)
                    relas_.append(rea_)
                    sub_heads.append(sub_headss_[j])
                    sub_tails.append(sub_tailss_[j])
                else:
                    for i in range(len(sub_headss_[j])):
                        sub_ = []
                        obj_ = []
                        rea_ = []
                        text_.append(texts_[j])
                        sub_.append(sub_headss_[j][i])
                        sub_.append(sub_tailss_[j][i])
                        obj_.append(obj_headss_[j][i])
                        obj_.append(obj_tailss_[j][i])
                        rea_.append(relass_[j][i])
                        sub_heads.append(sub_headss_[j])
                        sub_tails.append(sub_tailss_[j])
                        subs.append(sub_)
                        objs.append(obj_)
                        relas_.append(rea_)
                j += 1
            assert len(text_)==len(subs)
            assert len(text_)==len(objs)
            assert len(text_) == len(relas_)
            for i in range(len(text_)):
                examples.append(InputExample(text_[i],subs[i],objs[i],relas_[i],sub_heads[i],sub_tails[i]))
        else:
            text_ = data
            for i in range(len(text_)):
                examples.append(InputExample(text_[i]))
        return examples

    def _read(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            return data
