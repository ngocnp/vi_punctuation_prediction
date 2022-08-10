import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertForTokenClassification
import pickle
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer


def log_sum_exp_1vec(vec):  # shape(1,m)
    max_score = vec[0, np.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_mat(log_M, axis=-1):  # shape(n,m)
    return torch.max(log_M, axis)[0]+torch.log(torch.exp(log_M-torch.max(log_M, axis)[0][:, None]).sum(axis))


def log_sum_exp_batch(log_Tensor, axis=-1): # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0]+torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0],-1,1)).sum(axis))


class CrfBert(nn.Module):
    def __init__(self, bert_model, start_label_id, stop_label_id, num_labels, batch_size, device):
        super(CrfBert, self).__init__()
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.device = device

        self.bert = bert_model
        self.dropout = nn.Dropout(p=0.2)
        self.hidden2label = nn.Linear(self.bert.config.hidden_size, self.num_labels)

        self.transitions = nn.Parameter(torch.randn(self.num_labels))
        self.transitions.data[start_label_id, :] = -10000
        self.transitions.data[:, stop_label_id] = -10000

        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)

    def _forward_alg(self, feats):
        '''
        This also called alpha-recursion or forward recursion,  to calculate log_prob of all barX
        :param feats:
        :return:
        '''
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # alpha_recursion,  forward,  alpha(zt) = p(zt, bar_x_1:t)
        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000).to(self.device)
        log_alpha[:, 0, self.start_label_id] = 0
        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)
        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _get_bert_features(self, input_ids, segment_ids, input_mask):
        '''
        sentences => word embedding => lstm => MLP => feats
        :param input_ids:
        :param segment_ids:
        :param input_mask:
        :return:
        '''
        bert_seq_out, _ = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, output_all_encoded_layers=False)
        bert_seq_out = self.dropout(bert_seq_out)
        bert_feats = self.hidden2label(bert_seq_out)
        return bert_feats

    def _score_sentence(self, feats, label_ids):
        '''
        Gives the score of ta provided label sequence
        :param feats:
        :param label_ids:
        :return:
        '''
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size, self.num_labels, self.num_labels, self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0], 1)).to(self.device)
        for t in range(1, T):
            score = score + batch_transitions.gather(-1, (label_ids[:, t] + self.num_labels + label_ids[:, t-1]).view(-1, 1)) \
            + feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)
        return score

    def _viterbi_decode(self, feats):
        T = feats.shape[1]
        batch_size = feats.shape[0]

        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0

        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(self.device)
        for t in range(1, T):
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)
        path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)
        for t in range(T-2, -1, -1):
            path[:, t] = psi[:, t+1].gather(-1, path[:, t+1].view(-1, 1)).squeeze()
        return max_logLL_allz_allx, path

    def neg_log_likelihood(self, input_ids, segment_ids, input_mask, label_ids):
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)
        forward_score = self._forward_alg(bert_feats)
        gold_score = self._score_sentence(bert_feats, label_ids)
        return torch.mean(forward_score - gold_score)

    def forward(self, input_ids, segment_ids, input_mask):
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)
        score, label_seq_ids = self._viterbi_decode(bert_feats)
        return score, label_seq_ids
