# model += Parsing Infor Collecting Layer (PIC)

import matplotlib.pyplot as plt

import networkx as nx
from dgl import DGLGraph
import dgl
from src.functions.GAT import GAT

import numpy as np

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch
import torch.nn.functional as F

#from transformers.modeling_electra import ElectraModel, ElectraPreTrainedModel

from transformers import ElectraModel, RobertaModel

import transformers
if int(transformers.__version__[0]) <= 3:
    from transformers.modeling_roberta import RobertaPreTrainedModel
    from transformers.modeling_bert import BertPreTrainedModel
    from transformers.modeling_electra import ElectraModel, ElectraPreTrainedModel
else:
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
    from transformers.models.bert.modeling_bert import BertPreTrainedModel
    from transformers.models.electra.modeling_electra import ElectraPreTrainedModel

from src.functions.biattention import BiAttention, BiLinear



class RobertaForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, prem_max_sentence_length, hypo_max_sentence_length):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)

        # 입력 토큰에서 token1, token2가 있을 때 (index of token1, index of token2)를 하나의 span으로 보고 이에 대한 정보를 학습
        self.span_info_collect = SICModel1(config)

        # biaffine을 통해 premise와 hypothesis span에 대한 정보를 결합후 정규화
        self.gat = GATModel(config, prem_max_sentence_length, hypo_max_sentence_length) # 구묶음 + tag 정보 + klue-biaffine attention + bilistm + klue-bilinear classification

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        prem_span=None,
        hypo_span=None,
        prem_word_idxs=None,
        hypo_word_idxs=None,
    ):
        batch_size = input_ids.shape[0]
        discriminator_hidden_states = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last-layer hidden state
        # sequence_output: [batch_size, seq_length, hidden_size]
        sequence_output = discriminator_hidden_states[0]

        # span info collecting layer(SIC)
        h_ij = self.span_info_collect(sequence_output, prem_word_idxs, hypo_word_idxs)

        # parser info collecting layer(PIC)
        logits = self.gat(h_ij,
                          batch_size= batch_size,
                          prem_span=prem_span,hypo_span=hypo_span,)

        outputs = (logits, ) + discriminator_hidden_states[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            print("loss: "+str(loss))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)



class SICModel1(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_states, prem_word_idxs, hypo_word_idxs):
        # (batch, max_pre_sen, seq_len) @ (batch, seq_len, hidden) = (batch, max_pre_sen, hidden)
        prem_word_idxs = prem_word_idxs.squeeze(1)
        hypo_word_idxs = hypo_word_idxs.squeeze(1)

        prem = torch.matmul(prem_word_idxs, hidden_states)
        hypo = torch.matmul(hypo_word_idxs, hidden_states)

        return [prem, hypo]


class GATModel(nn.Module):
    def __init__(self, config, prem_max_sentence_length, hypo_max_sentence_length):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.prem_max_sentence_length = prem_max_sentence_length
        self.hypo_max_sentence_length = hypo_max_sentence_length
        self.num_labels = config.num_labels

        # 구문구조 종류
        depend2idx = {"None": 0};
        idx2depend = {0: "None"};
        for depend1 in ['IP', 'AP', 'DP', 'VP', 'VNP', 'S', 'R', 'NP', 'L', 'X']:
            for depend2 in ['CMP', 'MOD', 'SBJ', 'AJT', 'CNJ', 'None', 'OBJ', "UNDEF"]:
                depend2idx[depend1 + "-" + depend2] = len(depend2idx)
                idx2depend[len(idx2depend)] = depend1 + "-" + depend2
        self.depend2idx = depend2idx
        self.idx2depend = idx2depend
        self.depend_embedding = nn.Embedding(len(idx2depend), self.hidden_size, padding_idx=0).to("cuda")

        self.bi_lism_1 = nn.LSTM(input_size=100, hidden_size=self.hidden_size//2, num_layers=1, bidirectional=True)
        self.bi_lism_2 = nn.LSTM(input_size=100, hidden_size=self.hidden_size//2, num_layers=1, bidirectional=True)

        self.bilinear = BiLinear(self.hidden_size, self.hidden_size, self.num_labels)

    def forward(self, hidden_states, batch_size, prem_span, hypo_span):
        # hidden_states: [[batch_size, word_idxs, hidden_size], []]
        # span: [batch_size, max_sentence_length, max_sentence_length]
        # word_idxs: [batch_size, seq_length]
        # -> sequence_outputs: [batch_size, seq_length, hidden_size]

        prem_hidden_states= hidden_states[0]
        hypo_hidden_states= hidden_states[1]
        #print(prem_hidden_states.shape, hypo_hidden_states.shape, prem_span.shape, hypo_span.shape)

        # span: (batch, max_prem_len, 3) -> (batch, 3*max_prem_len, hidden_size)
        prem_gat = torch.tensor([], dtype=torch.long).to("cuda")
        hypo_gat = torch.tensor([], dtype=torch.long).to("cuda")

        for i, (p_span, h_span) in enumerate(zip(prem_span.tolist(), hypo_span.tolist())):
            # node랑 adj변경
            # 현재 같은 노드가 다시 등장
            # 같은 노드가 없도록 변경하고 그에 맞게 adj도 변경
            p_word = list(set([span[0] for span in p_span]+[span[1] for span in p_span]))
            p_dep = list(set([span[2] for span in p_span]))
            p_node = torch.cat((torch.index_select(prem_hidden_states[i], 0, torch.tensor(p_word).to("cuda")), self.depend_embedding(torch.tensor(p_dep).to("cuda"))))

            matrix_12 = torch.zeros([len(p_word), len(p_dep)], dtype=torch.int)
            matrix_21 = torch.zeros([len(p_dep), len(p_word)], dtype=torch.int)
            for j in range(0, len(p_span)):
                matrix_12[p_word.index(p_span[j][0])][p_dep.index(p_span[j][2])] = 1
                matrix_21[p_dep.index(p_span[j][2])][p_word.index(p_span[j][1])] = 1
            p_adj = torch.cat(
                (torch.cat((torch.zeros([len(p_word), len(p_word)], dtype=torch.int), matrix_12), dim=1),
                 torch.cat((matrix_21, torch.zeros([len(p_dep), len(p_dep)], dtype=torch.int)), dim=1)))

            # p_span_head = torch.tensor([span[0] for span in p_span]).to("cuda") #(max_prem_len)
            # p_span_tail = torch.tensor([span[1] for span in p_span]).to("cuda")
            # p_span_dep = torch.tensor([span[2] for span in p_span]).to("cuda")
            #
            # p_span_head = torch.index_select(prem_hidden_states[i], 0, p_span_head)  # (max_prem_len, hidden_size)
            # p_span_tail = torch.index_select(prem_hidden_states[i], 0, p_span_tail)
            # p_span_dep = self.depend_embedding(p_span_dep)
            #
            # p_node = torch.cat((p_span_head, p_span_dep, p_span_tail))  #(3*max_prem_len, hidden_size)
            #
            # zero_matrix = torch.zeros([self.prem_max_sentence_length, self.prem_max_sentence_length], dtype=torch.int)
            # one_diag_matrix = torch.diag(torch.ones([self.prem_max_sentence_length], dtype=torch.int))
            #
            # p_adj = torch.cat((torch.cat((zero_matrix, zero_matrix, zero_matrix),dim = 1), #(3*max_prem_len, 3*max_prem_len)
            #                    torch.cat((one_diag_matrix, zero_matrix, zero_matrix),dim = 1),
            #                    torch.cat((zero_matrix, one_diag_matrix, zero_matrix),dim = 1)))

            prem_g = self.numpy_to_graph(Adj=p_adj, type_graph='nx',node_features={'feat': p_node})
            # # Graph check
            # print(prem_g)
            # print(prem_g.nodes)
            # print([span[0] for span in p_span])
            # print([span[2] for span in p_span])
            # print([span[1] for span in p_span])
            # print(prem_g.edges)
            # print(prem_g.adj)
            # pos = nx.spring_layout(prem_g)  # pos = nx.nx_agraph.graphviz_layout(G)
            # nx.draw_networkx(prem_g, pos)
            # labels = nx.get_edge_attributes(prem_g, 'weight')
            # nx.draw_networkx_edge_labels(prem_g, pos, edge_labels=labels)
            # plt.show()

            prem_g= dgl.from_networkx(prem_g)
            #print(prem_g)
            prem_g = prem_g.to("cuda")

            prem_gat_func = GAT(prem_g,
                        in_dim=self.hidden_size,
                        hidden_dim=self.hidden_size,
                        out_dim=self.hidden_size,
                        num_heads=2).to("cuda")

            prem_gat_output = prem_gat_func(p_node)
            prem_gat = torch.cat((prem_gat, prem_gat_output.unsqueeze(0)))

            h_word = list(set([span[0] for span in h_span] + [span[1] for span in h_span]))
            h_dep = list(set([span[2] for span in h_span]))
            h_node = torch.cat((torch.index_select(hypo_hidden_states[i], 0, torch.tensor(h_word).to("cuda")),
                                self.depend_embedding(torch.tensor(h_dep).to("cuda"))))

            matrix_12 = torch.zeros([len(h_word), len(h_dep)], dtype=torch.int)
            matrix_21 = torch.zeros([len(h_dep), len(h_word)], dtype=torch.int)
            for j in range(0, len(h_span)):
                matrix_12[h_word.index(h_span[j][0])][h_dep.index(h_span[j][2])] = 1
                matrix_21[h_dep.index(h_span[j][2])][h_word.index(h_span[j][1])] = 1
            h_adj = torch.cat(
                (torch.cat((torch.zeros([len(h_word), len(h_word)], dtype=torch.int), matrix_12), dim=1),
                 torch.cat((matrix_21, torch.zeros([len(h_dep), len(h_dep)], dtype=torch.int)), dim=1)))


            hypo_g = self.numpy_to_graph(Adj=h_adj, type_graph='nx', node_features={'feat': h_node})

            hypo_g = dgl.from_networkx(hypo_g)
            hypo_g = hypo_g.to("cuda")

            hypo_gat_func = GAT(hypo_g,
                                in_dim=self.hidden_size,
                                hidden_dim=self.hidden_size,
                                out_dim=self.hidden_size,
                                num_heads=2).to("cuda")

            hypo_gat_output = hypo_gat_func(h_node)
            hypo_gat = torch.cat((hypo_gat, hypo_gat_output.unsqueeze(0)))

        # 여기부터 다시
        # bilstm
        # biaffine_outputs: [batch_size, max_prem_len,  100] -> [max_prem_len, batch_size, 100]
        # -> hidden_states: [batch_size, hidden_size]
        print(prem_gat.shape)
        print(hypo_gat.shape)
        exit()
        prem_gat = prem_gat.transpose(0,1)
        hypo_gat = hypo_gat.transpose(0,1)

        prem_bilstm_outputs, prem_states = self.bi_lism_1(prem_gat)
        hypo_bilstm_outputs, hypo_states = self.bi_lism_2(hypo_gat)


        prem_hidden_states = prem_states[0].transpose(0, 1).contiguous().view(batch_size, -1)
        hypo_hidden_states = hypo_states[0].transpose(0, 1).contiguous().view(batch_size, -1)

        outputs = self.bilinear(prem_hidden_states, hypo_hidden_states)

        return outputs

    def numpy_to_graph(self, Adj, type_graph='dgl', node_features=None):
        '''Convert numpy arrays to graph

        Parameters
        ----------
        A : mxm array
            Adjacency matrix
        type_graph : str
            'dgl' or 'nx'
        node_features : dict
            Optional, dictionary with key=feature name, value=list of size m
            Allows user to specify node features

        Returns

        -------
        Graph of 'type_graph' specification
        '''
        np_Adj = Adj.cpu().detach().numpy()
        G = nx.from_numpy_array(np_Adj).to_directed()

        if node_features != None:
            for n in G.nodes():
                for k, v in node_features.items():
                    G.nodes[n][k] = v[n]

        if type_graph == 'nx':
            return G

        G = G.to_directed()

        if node_features != None:
            node_attrs = list(node_features.keys())
        else:
            node_attrs = []

        g = dgl.from_networkx(G, node_attrs=node_attrs, edge_attrs=['weight'])
        return g