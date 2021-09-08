# + sentence embedding

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch

from transformers.file_utils import is_torch_available
device = 'cuda' if is_torch_available() else 'cpu'

from transformers import RobertaModel

import transformers
if int(transformers.__version__[0]) <= 3:
    from transformers.modeling_roberta import RobertaPreTrainedModel
    from transformers.modeling_bert import BertPreTrainedModel
else:
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
    from transformers.models.bert.modeling_bert import BertPreTrainedModel



class RobertaForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 lambd = 0.1,
                 ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)

        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1

        self.debias = nn.Linear(2 * config.hidden_size, config.num_labels)
        self.classifier = nn.Linear(num_vectors_concatenated * config.hidden_size, config.num_labels)
        self.linear_1 = nn.Linear(num_vectors_concatenated * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 일반화된 정보를 사용
        self.linear_2 = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        prem_input_ids=None,
        prem_attention_mask=None,
        prem_token_type_ids=None,
        prem_position_ids=None,
        prem_head_mask=None,
        prem_inputs_embeds=None,
        hypo_input_ids=None,
        hypo_attention_mask=None,
        hypo_token_type_ids=None,
        hypo_position_ids=None,
        hypo_head_mask=None,
        hypo_inputs_embeds=None,
        prem_token_length=None,
        hypo_token_length=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

        """
        # 보통 여기까지
        # discriminator_hidden_states : (last_hidden_state, pooler_output
        # config에 추가하여 ## 참조: https://huggingface.co/transformers/model_doc/roberta.html
        # discriminator_hidden_states : (last_hidden_state, pooler_output, all hidden states, past_key_values, all attentions, all cross_attention)
        prem_discriminator_hidden_states = self.roberta(
            input_ids=prem_input_ids,
            attention_mask=prem_attention_mask,
        )

        hypo_discriminator_hidden_states = self.roberta(
            input_ids=hypo_input_ids,
            attention_mask=hypo_attention_mask,
        )

        # last-layer hidden state
        # sequence_output: [batch_size, seq_length, hidden_size]
        prem_sequence_output = prem_discriminator_hidden_states[0]
        hypo_sequence_output = hypo_discriminator_hidden_states[0]

        def mean_pooing(att_mask, tok_emb):
            input_mask_expanded = att_mask.unsqueeze(-1).expand(tok_emb.size()).float()
            sum_embeddings = torch.sum(tok_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            sen_emb = sum_embeddings / sum_mask
            return sen_emb

        # sentence_embedding = [batch_size, hidden_size]
        prem_sentence_embedding = mean_pooing(prem_attention_mask, prem_sequence_output)
        hypo_sentence_embedding = mean_pooing(hypo_attention_mask, hypo_sequence_output)

        # rep_a, rep_b: [batch_size, hidden_size]
        rep_a, rep_b = prem_sentence_embedding, hypo_sentence_embedding

        vectors_concat = []

        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        # rep_a, rep_b: [batch_size, len(vectors_concat)*hidden_size]
        features = torch.cat(vectors_concat, 1)
        #logits = self.classifier(features)
        features = self.dropout(features)

        # features: [batch_size, hidden_size]
        features = self.linear_1(features)
        features = torch.tanh(features)
        features = self.dropout(features)

        # logits: [batch_size, num_labels]
        logits = self.linear_2(features)

        ##################### sub-model #############################
        ## attention: [12, batch, num_heads, seq_length, seq_length]
        ## attention[-1]: [batch, num_heads, seq_length, seq_length]
        #prem_attention = prem_discriminator_hidden_states[2][-1]
        #hypo_attention = hypo_discriminator_hidden_states[2][-1]

        # attention: [batch, num_heads, seq_length, seq_length] -> [batch, seq_length, seq_length]

        outputs = (logits,)

        if labels is not None:

            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            print("loss" + str(loss))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits






