from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch

from transformers import RobertaModel

import transformers
if int(transformers.__version__[0]) <= 3:
    from transformers.modeling_roberta import RobertaPreTrainedModel
    from transformers.modeling_bert import BertPreTrainedModel
    from transformers.modeling_electra import ElectraModel, ElectraPreTrainedModel
else:
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
    from transformers.models.bert.modeling_bert import BertPreTrainedModel
    from transformers.models.electra.modeling_electra import ElectraPreTrainedModel

class RobertaForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)

        self.linear_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 일반화된 정보를 사용
        self.linear_2 = nn.Linear(config.hidden_size, config.num_labels)

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
    ):

        discriminator_hidden_states = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last-layer hidden state
        # sequence_output: [batch_size, seq_length, hidden_size]
        sequence_output = discriminator_hidden_states[0]

        # sequence_output: [batch_size, hidden_size]
        sequence_output = sequence_output[:, 0, :]  # CLS token
        sequence_output = self.dropout(sequence_output)

        # sequence_output: [batch_size, hidden_size]
        sequence_output = self.linear_1(sequence_output)
        sequence_output = torch.tanh(sequence_output)
        sequence_output = self.dropout(sequence_output)

        # logits: [batch_size, num_labels]
        logits = self.linear_2(sequence_output)

        outputs = (logits,) + discriminator_hidden_states[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            print("loss: " + str(loss))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
