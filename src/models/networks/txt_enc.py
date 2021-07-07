import torch
import torch.nn as nn
import torch.nn.init
from transformers import BertModel

class Transformer(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.sent = self.cfg.sent_embedding
    self.bert = BertModel.from_pretrained(self.cfg.bert_pretrain)
    self.embedding_dim = self.bert.config.to_dict()['hidden_size']
    
  def forward(self, text, mask=None, type_id=None):
    # text = [batch size, sent len]
    embedded = self.bert(input_ids=text,attention_mask=mask,token_type_ids=type_id)[0]
    words = embedded
    
    if self.sent == "cls":
      sent = embedded[:, 0, :]
      
    elif self.sent == "mean":
      embedded = embedded * mask.unsqueeze(-1)
      sent = embedded.mean(dim=1)
      
    else:
      raise ValueError("cls or mean")
      
    return words, sent