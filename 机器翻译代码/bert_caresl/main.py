import argparse
import os
import torch
from common import seed_everything
from transformers import BertConfig,BertTokenizer,BertModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from model import Casrel
from processor import REProcessor,collate_fn,collate_fn1
from dataset import convert_examples_to_features
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import tqdm


def load_and_cache_examples(args, tokenizer, data_type=None):
    processor = REProcessor()
    if data_type=='train':
        examples = processor.get_train_examples(args.data_dir)
    elif data_type=='dev':
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)
    dataset = convert_examples_to_features(args, examples=examples,tokenizer=tokenizer, data_type=data_type)
    return dataset

def train(args, train_data, model, tokenizer):
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,collate_fn=collate_fn)
    no_decay = ["bias", "LayerNorm.weight"]
    parameterss = list(model.bert_encoder.named_parameters())+list(model.sub_heads_linear.named_parameters())+list(model.sub_tails_linear.named_parameters())+\
    list(model.obj_heads_linear.named_parameters())+list(model.obj_tails_linear.named_parameters())
    optimizer_grouped_parameters = [
        {"params": [p for n, p in parameterss if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in parameterss if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.learning_rate},
    ]
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    # define the loss function
    def loss(gold, pred, mask):
        los = F.binary_cross_entropy(pred, gold, reduction='none')
        if los.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        los = torch.sum(los * mask) / torch.sum(mask)
        return los
    for epoch in range(int(args.num_train_epochs)):
        steps = 0
        losses = 0
        for step, batch in tqdm.tqdm(enumerate(train_dataloader)):
            steps += 1
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            data = {"token_ids": batch[0], "mask": batch[1], "sub_head":batch[-3],"sub_tail":batch[-2]}
            pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails = model(data)
            pred_sub_heads = pred_sub_heads.squeeze(-1)
            pred_sub_tails = pred_sub_tails.squeeze(-1)
            gold_obj_heads = batch[5].permute(0,2,1).contiguous().float()
            gold_obj_tails = batch[6].permute(0,2,1).contiguous().float()
            sub_heads_loss = loss(batch[3].float(), pred_sub_heads, data['mask'])
            sub_tails_loss = loss(batch[4].float(), pred_sub_tails, data['mask'])
            obj_heads_loss = loss(gold_obj_heads, pred_obj_heads, data['mask'])
            obj_tails_loss = loss(gold_obj_tails, pred_obj_tails, data['mask'])
            total_loss = (sub_heads_loss + sub_tails_loss) + (obj_heads_loss + obj_tails_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losses += total_loss
        print('第{}epoch的loss为{}'.format(epoch, losses / float(steps)))
        torch.cuda.empty_cache()
        if (epoch+1)%20==0:
          path = os.path.join(args.output_dir, str(epoch+1) + 'pt')
          torch.save(model.state_dict(), path)
def aa(pred_sub_heads,pred_sub_tails,start_mask,end_mask):
  sub_heads, sub_tails = np.where(pred_sub_heads.cpu() > 0.4)[0], np.where(pred_sub_tails.cpu() > 0.4)[0]
  # print(sub_heads)
  # print(sub_tails)
  pred_sub_heads1 = torch.masked_select(pred_sub_heads,start_mask==1)
  pred_sub_heads1 = pred_sub_heads1.reshape(-1,1)
  pred_sub_tails1 = torch.masked_select(pred_sub_tails,end_mask==1)
  pred_sub_tai1s1 = pred_sub_tails1.reshape(-1,1)
  sub_heads1, sub_tails1 = np.where(pred_sub_heads1.cpu() > 0.4)[0], np.where(pred_sub_tails1.cpu() > 0.4)[0]
  subjects = []
  subjects1 = []
  for sub_head in sub_heads:
    sub_tail = sub_tails[sub_tails >= sub_head]
    if len(sub_tail) > 0:
      sub_tail = sub_tail[0]
      subjects.append((sub_head,sub_tail))
  pre_tail = -1
  for sub_head1 in sub_heads1:
    # if sub_head1<=pre_tail:
    #   continue
    sub_tail1 = sub_tails1[sub_tails1 >= sub_head1]
    if len(sub_tail1) > 0:
      sub_tail1 = sub_tail1[0]
      pre_tail = sub_tail1
      subjects1.append((sub_head1,sub_tail1))
  return subjects,subjects1
def predict(args, test_data, model, tokenizer):
    model.eval()
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=1,collate_fn=collate_fn1)
    rel_num = len(args.id2rel)
    results = []
    count  = 0
    for step, batch in tqdm.tqdm(enumerate(test_dataloader)):
        batch = tuple(t.to(args.device) for t in batch)
        data = {"token_ids": batch[0], "mask": batch[1]}
        with torch.no_grad():
            token_ids = data['token_ids']
            mask = data['mask']
            seq_len = mask.shape
            
            start_mask = batch[-2].permute(1,0).contiguous()
            end_mask = batch[-1].permute(1,0).contiguous()
            encoded_text = model.get_encoded_text(token_ids, mask)
            pred_sub_heads, pred_sub_tails = model.get_subs(encoded_text)
            # (seq_len,1)
            pred_sub_tails = torch.mul(pred_sub_tails.squeeze(0),end_mask)
            pred_sub_heads = torch.mul(pred_sub_heads.squeeze(0),start_mask)
            # assert pred_sub_heads.shape==(seq_len,1)
            # print("seq_len:{}".format(pred_sub_heads.shape))
            # print("seq_len:{}".format(start_mask.shape))
            # print(pred_sub_heads)
            # print(pred_sub_tails)
            sub_heads, sub_tails = np.where(pred_sub_heads.cpu() > 0.5)[0], np.where(pred_sub_tails.cpu() > 0.5)[0]
            # print(sub_heads)
            # print(sub_tails)
            pred_sub_heads1 = torch.masked_select(pred_sub_heads,start_mask==1)
            pred_sub_heads1 = pred_sub_heads1.reshape(-1,1)
            pred_sub_tails1 = torch.masked_select(pred_sub_tails,end_mask==1)
            pred_sub_tai1s1 = pred_sub_tails1.reshape(-1,1)
            sub_heads1, sub_tails1 = np.where(pred_sub_heads1.cpu() > 0.5)[0], np.where(pred_sub_tails1.cpu() > 0.5)[0]
            subjects = []
            subjects1 = []
            # pre_tail = -1
            for sub_head in sub_heads:
              # if sub_head<=pre_tail:
              #   continue
              sub_tail = sub_tails[sub_tails >= sub_head]
              if len(sub_tail) > 0:
                sub_tail = sub_tail[0]
                # pre_tail = sub_tail
                subjects.append((sub_head,sub_tail))
            # pre_tail1 = -1
            for sub_head1 in sub_heads1:
              # if sub_head1<=pre_tail1:
              #   continue
              sub_tail1 = sub_tails1[sub_tails1 >= sub_head1]
              if len(sub_tail1) > 0:
                sub_tail1 = sub_tail1[0]
                # pre_tail1 = sub_tail1
                subjects1.append((sub_head1,sub_tail1))

            # if subjects1==[]:
            #   subjects,subjects1 = aa(pred_sub_heads,pred_sub_tails,start_mask,end_mask)
            if subjects1==[]:
              count += 1
            # print('-----')
            # print(subjects1)
            if subjects:
                triple_list = []
                # [subject_num, seq_len, bert_dim]
                repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
                # [subject_num, 1, seq_len]
                sub_head_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
                sub_tail_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
                for subject_idx, subject in enumerate(subjects):
                    sub_head_mapping[subject_idx][0][subject[0]] = 1
                    sub_tail_mapping[subject_idx][0][subject[1]] = 1
                sub_tail_mapping = sub_tail_mapping.to(repeated_encoded_text)
                sub_head_mapping = sub_head_mapping.to(repeated_encoded_text)
                pred_obj_heads, pred_obj_tails = model.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping,
                                                                                 repeated_encoded_text)
                pred_obj_heads = pred_obj_heads.permute(0,2,1).contiguous()
                pred_obj_tails = pred_obj_tails.permute(0,2,1).contiguous()
                for subject_idx, subject in enumerate(subjects1):
                    for i in range(pred_obj_heads.shape[1]):
                        pred_obj_head = torch.masked_select(pred_obj_heads[subject_idx,i].reshape(-1,1),start_mask==1)
                        pred_obj_tail = torch.masked_select(pred_obj_tails[subject_idx,i].reshape(-1,1),end_mask==1)
                        pred_obj_head = pred_obj_head.reshape(-1,1)
                        pred_obj_tail = pred_obj_tail.reshape(-1, 1)
                        obj_heads, obj_tails = np.where(pred_obj_head.cpu() > 0.5)[0], np.where(pred_obj_tail.cpu() > 0.5)[0]
                        # pre_tail2 = -1
                        for obj_head in obj_heads:
                          # if obj_head<=pre_tail2:
                          #   continue
                          obj_tail = obj_tails[obj_tails >= obj_head]
                          if len(obj_tail) > 0:
                            obj_tail = obj_tail[0]
                            # pre_tail2 = obj_tail
                            rel = args.id2rel[int(i)]
                            triple_list.append((subject,rel,(obj_head,obj_tail)))
                results.append(triple_list)

            else:
                results.append([])
            # print(triple_list)
            # if step==9:
            #   break
    xx = 0
    for x in results:
      if x==[]:
        xx += 1
    print('count:{}'.format(xx))
    with open(args.output_dir+'result.pk','wb') as f:
        pickle.dump(results, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='datasets', type=str, help="The input data dir.", )
    parser.add_argument("--model_path", default='bert-pretrain/', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default='outputs', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.", )
    parser.add_argument("--tokenizer_name", default="bert-pretrain/", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--config_name", default="bert-pretrain/bert_config.json", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--train_batch_size", default=12, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.00005, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=100, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--max_length", type=int, default=512, help="random seed for initialization")
    parser.add_argument("--num_rel", type=int, default=5, help="random seed for initialization")
    args = parser.parse_args()
    device = torch.device('cuda')
    args.device = device
    seed_everything(args.seed)
    rel_list = REProcessor().get_rels()
    args.id2rel = {i:r for i,r in enumerate(rel_list)}
    args.rel2id = {r:i for i,r in args.id2rel.items()}
    config = BertConfig.from_pretrained(args.config_name)
    config.rel_num = args.num_rel
    config.model_path = args.model_path
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name,do_lower_case=False)
    model = Casrel(config)
    model.to(args.device)
    if args.do_train:
        train_data = load_and_cache_examples(args, tokenizer, data_type='train')
        train(args, train_data, model, tokenizer)
    if args.do_predict:
        model = Casrel(config)
        checkpoints = '/content/drive/My Drive/bert_caresl/outputs/100pt'
        print(checkpoints)
        model.load_state_dict(torch.load(checkpoints))
        model.to(args.device)
        test_data = load_and_cache_examples(args, tokenizer, data_type='test')
        predict(args, test_data, model, tokenizer)

if __name__=='__main__':
    main()