from transformers.models.electra.modeling_electra import ElectraModel, ElectraPreTrainedModel, ElectraForMaskedLM
from transformers.modeling_utils import PreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

def swish(x):
    return x * torch.sigmoid(x)


def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


ACT2FN = {
    "relu": F.relu,
    "swish": swish,
    "gelu": gelu,
    "tanh": torch.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
}

def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(activation_string, list(ACT2FN.keys())))

class ElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_sequence_output):
        discriminator_sequence_output = discriminator_sequence_output[:, 1:, :]
        hidden_states = self.dense(discriminator_sequence_output)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze()

        return logits


class ElectraRMD(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, disc_hid):
        super().__init__()

        self.fc1 = nn.Linear(disc_hid, disc_hid)
        self.fc2 = nn.Linear(disc_hid, 1)

    def forward(self, discriminator_sequence_output):
        input_ = discriminator_sequence_output[:, 0, :]
        # input_ = torch.squeeze(discriminator_sequence_output, dim=1)

        fc1_out = F.relu(self.fc1(input_))
        fc2_out = self.fc2(fc1_out)

        return fc2_out

class ElectraForPreTraining(ElectraPreTrainedModel):
    def __init__(self, config, output_size = 6):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
        self.rmd_predictions = ElectraRMD(config.hidden_size)
        self.init_weights()
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None
    ):
        discriminator_hidden_states = self.electra(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, position_ids = position_ids, inputs_embeds = inputs_embeds, output_attentions = True)
        discriminator_sequence_output = discriminator_hidden_states[0]
        attentions = discriminator_hidden_states[-1]
        last_attention = attentions[-1]
        # rmd_output = self.rmd_predictions(discriminator_sequence_output)
        logits = self.discriminator_predictions(discriminator_sequence_output)
        rtd_loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_loss = active_loss[:, 1:]
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1] - 1)[active_loss]
                active_labels = labels[active_loss]
                rtd_loss = loss_fct(active_logits, active_labels.float())
            else:
                rtd_loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1] - 1), labels.float())
        
        return discriminator_sequence_output[:, 0, :], rtd_loss, logits, last_attention
    

class ElectraForLanguageModelingModel(PreTrainedModel):
    def __init__(self, config, vocab, output_size: int = 100, random_generator = False, **kwargs):
        super().__init__(config)
        self.generator_model = ElectraForMaskedLM(config)
        self.discriminator_model = ElectraForPreTraining(config, output_size)
        self.vocab = vocab

        
        self.random_generator = random_generator
        print(f"IN MODEL: RANDOM GENERATOR: {self.random_generator}")
    
    def tie_generator_and_discriminator_embeddings(self):
        gen_embeddings = self.generator_model.electra.embeddings
        disc_embeddings = self.discriminator_model.electra.embeddings

        # tie word, position and token_type embeddings
        gen_embeddings.word_embeddings.weight = disc_embeddings.word_embeddings.weight
        gen_embeddings.position_embeddings.weight = disc_embeddings.position_embeddings.weight
        gen_embeddings.token_type_embeddings.weight = disc_embeddings.token_type_embeddings.weight


    def forward(self, inputs, labels, attention_mask = None):
        d_inputs = inputs.clone()
        vocab_size = len(self.vocab)

        sample_tokens = torch.zeros_like(d_inputs)
        g_out = self.generator_model(inputs, labels = labels, attention_mask = attention_mask)
        temperature = 1.0
        sample_probs = torch.softmax(g_out[1] / temperature, dim = -1, dtype = torch.float64)
        sample_probs = sample_probs.view(-1, vocab_size)

        # top10_percent_indices = torch.topk(1 - sample_probs, int(0.1 * vocab_size), dim = -1)[1]
        # complement_probs = torch.ones_like(sample_probs) * 1e-4
        # complement_probs.scatter_(1, top10_percent_indices, 1 - sample_probs.gather(1, top10_percent_indices))
        # sample_tokens = torch.multinomial(complement_probs, 1).view(-1)
        # sample_tokens = sample_tokens.view(d_inputs.shape[0], -1)


        sample_tokens = torch.multinomial(1 - sample_probs, 1).view(-1)
        sample_tokens = sample_tokens.view(d_inputs.shape[0], -1)


        if self.random_generator:
            sample_tokens = torch.randint(4, vocab_size - 1, sample_tokens.shape)
            sample_tokens = sample_tokens.to(d_inputs.device)


        mask = labels.ne(-100)

        d_inputs[mask] = sample_tokens[mask]

        correct_preds = sample_tokens == labels
        d_labels = mask.long()
        d_labels[correct_preds] = 0

        cls_token = (torch.ones(d_inputs.shape[0], 1) * self.vocab.sos_index).to(d_inputs.device)
        d_inputs = torch.concat([cls_token, d_inputs], dim = 1).long()
        attention_mask = torch.concat([torch.ones(attention_mask.shape[0], 1).to(attention_mask.device), attention_mask], dim = 1).long()

        d_out = self.discriminator_model(d_inputs, labels = d_labels, attention_mask = attention_mask)

        rmd_output = d_out[0]
        rtd_loss = d_out[1]
        d_scores = d_out[2]
        last_attention = d_out[3]
        g_loss = g_out[0]
        g_scores = g_out[1]

        return rmd_output, rtd_loss, d_scores, g_loss, g_scores, last_attention
        # return rmd_output, rtd_loss, d_scores

    

