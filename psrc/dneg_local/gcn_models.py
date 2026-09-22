from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Dict, Any
from transformers import (PreTrainedModel,
                          BertConfig,
                          BertModel,
                          AutoModel,
                          AutoConfig,
                          AutoTokenizer,
                          PreTrainedTokenizer,
                          AutoModelForTokenClassification,
                          TrainingArguments,
                          Trainer,
                          DataCollatorForTokenClassification,
                          PreTrainedTokenizerBase, BertPreTrainedModel)
from transformers.modeling_outputs import TokenClassifierOutput
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
import torch
import os


from torch_geometric.utils import add_self_loops

#torch.manual_seed(42)
#torch.cuda.manual_seed(42)
EMPTY_SPECIAL_TOKEN = "[EMPTY]"
BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))


class BERTWithGCNForTokenClassification(BertPreTrainedModel):
    config_class = BertConfig

    def __init__(self, config, pos_vocab_size=None, dep_vocab_size=None, merge_strategy='average'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = AutoModel.from_config(config)
        self.relu = torch.nn.ReLU() # Added activation as discussed previously
        self.gelu = torch.nn.GELU()

        self.pos_emb = torch.nn.Embedding(pos_vocab_size, int(config.hidden_size * 0.1))
        self.dep_emb = torch.nn.Embedding(dep_vocab_size, int(config.hidden_size * 0.1))
        self.ln_pre_gcn = torch.nn.LayerNorm(config.hidden_size + 2 * int(config.hidden_size * 0.1), eps=1e-5)
        # self.ln_word_emb = torch.nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ln_gcn = torch.nn.LayerNorm(config.hidden_size, eps=1e-5)

        self.gcn = GCNConv(in_channels=config.hidden_size + 2 * int(config.hidden_size * 0.1),
                           out_channels=config.hidden_size#  + 2 * int(config.hidden_size * 0.1)
                           )
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, # + 2 * int(config.hidden_size * 0.1),
                                          config.num_labels)

        self.post_init()

        try:
            torch.nn.init.xavier_uniform_(self.gcn.lin.weight)
            if self.gcn.bias is not None:
                torch.nn.init.zeros_(self.gcn.bias)
        except AttributeError:
             print("GCN linear layer or bias not found for explicit init.")

        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.zeros_(self.classifier.bias)
        with torch.no_grad():
            self.gcn.lin.weight.data.mul_(0.01)
            self.classifier.weight.data.mul_(0.01)



    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            pos_ids: Optional[torch.Tensor] = None,
            dep_ids: Optional[torch.Tensor] = None,
            edge_index: Optional[List[torch.Tensor]] = None,
            word_ids: Optional[list] = None, # list of lists
            labels: Optional[torch.Tensor] = None, # flattened labels (total_words)
            word_count: Optional[List[int]] = None, # list of num_words per sentence
            # ----------------------------------------------
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if word_ids is None or word_count is None or edge_index is None:
             raise ValueError("word_ids, word_count, and edge_index are required inputs.")

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]

        pos_embed = self.pos_emb(pos_ids)  # Shape: (batch_size, max_seq_length, 50)
        dep_embed = self.dep_emb(dep_ids)  # Shape: (batch_size, max_seq_length, 50)

        # Combine BERT output with POS and DEP embeddings
        sequence_output = torch.cat([sequence_output, pos_embed, dep_embed], dim=-1)
        sequence_output = self.ln_pre_gcn(sequence_output)

        if torch.isnan(sequence_output).any():
            print(sequence_output)
            raise RuntimeError("NaNs after bert layer")

        batch_size, seq_length, hidden_size = sequence_output.shape
        batch_word_embeddings = []
        for i in range(batch_size):
            # Use a set to track word IDs seen *for the first time* in this sentence
            seen_word_ids_in_sentence = set()
            word_embeddings_list = []
            for j in range(word_count[i]):
                if j in word_ids[i]:
                    word_embeddings_list.append(sequence_output[i][word_ids[i].index(j)])
                else:
                    word_embeddings_list.append(sequence_output[i][0])

            batch_word_embeddings.append(torch.stack(word_embeddings_list))


        # Create pytorch_geometric Data objects for the batch
        offset = 0
        edge_indices = [[], []]
        for i in range(batch_size):
            edge_index_sent, _ = add_self_loops(edge_index[i], num_nodes=batch_word_embeddings[i].size(0))
            edge_indices[0].extend(edge_index_sent[0] + offset)
            edge_indices[1].extend(edge_index_sent[1] + offset)
            offset += batch_word_embeddings[i].size(0)

        gcn_out = torch.cat(batch_word_embeddings, dim=0)
        edge_index = torch.tensor(edge_indices).to(self.device)
        deg = degree(edge_index[1], num_nodes=gcn_out.size(0)).clamp(min=1)
        edge_weight = 1.0 / deg[edge_index[1]]

        if torch.isnan(gcn_out).any():
            print(gcn_out)
            print("NaNs found in word embeddings after scatter/LayerNorm:", gcn_out)
            raise RuntimeError("NaNs found in word embeddings after scatter/LayerNorm")

        # print(edge_index.shape, edge_index.min(), edge_index.max())
        # print(edge_index)
        # print(gcn_out.shape)
        # GCN forward pass
        gcn_out = self.gcn(gcn_out, edge_index, edge_weight=edge_weight)
        gcn_out = self.gelu(gcn_out)
        gcn_out = self.ln_gcn(gcn_out)
        gcn_out = self.dropout(gcn_out)

        if torch.isnan(gcn_out).any():
            mean = gcn_out.mean(dim=-1, keepdim=True)
            var = gcn_out.var(dim=-1, keepdim=True, unbiased=False)
            print(var.min(), mean.min(), var.max(), mean.max(), gcn_out.shape)
            print("NaNs found in GCN output after LayerNorm:", gcn_out)
            raise RuntimeError("NaNs found in GCN output after LayerNorm")
        if torch.isinf(gcn_out).any():
            mean = gcn_out.mean(dim=-1, keepdim=True)
            var = gcn_out.var(dim=-1, keepdim=True, unbiased=False)
            print(var.min(), mean.min(), var.max(), mean.max(), gcn_out.shape)
            print("Inf found in GCN output after LayerNorm:", gcn_out)
            raise RuntimeError("NaNs found in GCN output after LayerNorm")

        # Classifier
        logits = self.classifier(gcn_out)
        # print(logits)
        if torch.isnan(logits).any():
            print("NaNs found in GCN output after LayerNorm:", logits)
            raise RuntimeError("NaNs found in GCN output after LayerNorm")
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # Assuming labels are flattened to (total_num_words,)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(logits.view(-1, self.num_labels), labels.view(-1))
                print(loss)
                raise RuntimeError("NaNs or Infs found in Loss")

        if not return_dict:
             raise Exception("return_dict=False is not fully implemented in the restructured code.")

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BERTWithGATForTokenClassificationResidual(BertPreTrainedModel):
    config_class = BertConfig

    def __init__(self, config, pos_vocab_size=None, dep_vocab_size=None, merge_strategy='average'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = AutoModel.from_config(config)
        self.relu = torch.nn.ReLU() # Added activation as discussed previously
        self.gelu = torch.nn.GELU()

        self.pos_emb = torch.nn.Embedding(pos_vocab_size, int(config.hidden_size * 0.1))
        self.dep_emb = torch.nn.Embedding(dep_vocab_size, int(config.hidden_size * 0.1))
        self.ln_pre_gcn = torch.nn.LayerNorm(config.hidden_size + 2 * int(config.hidden_size * 0.1), eps=1e-5)
        self.reduction_layer = torch.nn.Linear(config.hidden_size + 2 * int(config.hidden_size * 0.1),
                                               128)
        self.ln_reduction = torch.nn.LayerNorm(128, eps=1e-5)

        self.cat_norm = torch.nn.LayerNorm(128, eps=1e-5)
        # self.ln_word_emb = torch.nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ln_gcn = torch.nn.LayerNorm(128, eps=1e-5)

        self.gcn = GATv2Conv(
            in_channels=128,
            out_channels=128,
            add_self_loops=True,
            heads=4,  # Multi-head attention
            dropout=0.1,  # Attention dropout
            concat=False,
            residual=True# Concatenate heads
        )
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(128, # + 2 * int(config.hidden_size * 0.1),
                                          config.num_labels)

        self.post_init()

        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.zeros_(self.classifier.bias)

        torch.nn.init.xavier_uniform_(self.reduction_layer.weight)
        torch.nn.init.zeros_(self.reduction_layer.bias)

        torch.nn.init.xavier_uniform_(self.pos_emb.weight, gain=0.01)
        torch.nn.init.xavier_uniform_(self.dep_emb.weight, gain=0.01)

        for name, param in self.gcn.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

        with torch.no_grad():
            self.classifier.weight.data.mul_(0.1)
            self.reduction_layer.weight.data.mul_(0.1)
            self.pos_emb.weight.data.mul_(0.1)
            self.dep_emb.weight.data.mul_(0.1)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            pos_ids: Optional[torch.Tensor] = None,
            dep_ids: Optional[torch.Tensor] = None,
            edge_index: Optional[List[torch.Tensor]] = None,
            word_ids: Optional[list] = None, # list of lists
            labels: Optional[torch.Tensor] = None, # flattened labels (total_words)
            word_count: Optional[List[int]] = None, # list of num_words per sentence
            # ----------------------------------------------
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if word_ids is None or word_count is None or edge_index is None:
             raise ValueError("word_ids, word_count, and edge_index are required inputs.")

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]

        pos_embed = self.pos_emb(pos_ids)  # Shape: (batch_size, max_seq_length, 50)
        dep_embed = self.dep_emb(dep_ids)  # Shape: (batch_size, max_seq_length, 50)

        # Combine BERT output with POS and DEP embeddings
        sequence_output = torch.cat([sequence_output, pos_embed, dep_embed], dim=-1)
        sequence_output = self.ln_pre_gcn(sequence_output)
        sequence_output = self.relu(sequence_output)
        sequence_output = self.ln_reduction(self.relu(self.reduction_layer(sequence_output)))

        if torch.isnan(sequence_output).any():
            print(sequence_output)
            raise RuntimeError("NaNs after bert layer")

        batch_size, seq_length, hidden_size = sequence_output.shape
        batch_word_embeddings = []
        for i in range(batch_size):
            # Use a set to track word IDs seen *for the first time* in this sentence
            seen_word_ids_in_sentence = set()
            word_embeddings_list = []
            for j in range(word_count[i]):
                if j in word_ids[i]:
                    word_embeddings_list.append(sequence_output[i][word_ids[i].index(j)])
                else:
                    word_embeddings_list.append(sequence_output[i][0])

            batch_word_embeddings.append(torch.stack(word_embeddings_list))


        # Create pytorch_geometric Data objects for the batch
        offset = 0
        edge_indices = [[], []]
        for i in range(batch_size):
            """edge_index_sent, _ = add_self_loops(edge_index[i], num_nodes=batch_word_embeddings[i].size(0))
            edge_indices[0].extend(edge_index_sent[0] + offset)
            edge_indices[1].extend(edge_index_sent[1] + offset)"""
            edge_indices[0].extend(edge_index[i][0] + offset)
            edge_indices[1].extend(edge_index[i][1] + offset)
            offset += batch_word_embeddings[i].size(0)

        gcn_in = torch.cat(batch_word_embeddings, dim=0)
        edge_index = torch.tensor(edge_indices, device=gcn_in.device)
        print("edge_index:", edge_index.shape, edge_index)
        print("unique nodes:", torch.unique(edge_index).numel())
        print("nodeshape:", gcn_in.shape)
        deg = degree(edge_index[1], num_nodes=gcn_in.size(0)).clamp(min=1)
        edge_weight = 1.0 / deg[edge_index[1]]

        gcn_in = self.cat_norm(gcn_in)
        if torch.isnan(gcn_in).any():
            print(gcn_in)
            print("NaNs found in word embeddings after scatter/LayerNorm:", gcn_in)
            raise RuntimeError("NaNs found in word embeddings after scatter/LayerNorm")

        # print(edge_index.shape, edge_index.min(), edge_index.max())
        # print(edge_index)
        # print(gcn_out.shape)
        # GCN forward pass
        num_nodes = gcn_in.size(0)
        if edge_index.min() < 0 or edge_index.max() >= num_nodes:
            raise ValueError(
                f"edge_index contains out-of-bounds node indices: min={edge_index.min()}, max={edge_index.max()}, num_nodes={num_nodes}")
        edge_index = torch.unique(edge_index, dim=1)
        gcn_out = self.relu(self.gcn(gcn_in, edge_index))

        if torch.isnan(gcn_out).any():
            print("in", gcn_in.shape, gcn_in.min(), gcn_in.max())
            mean = gcn_out.mean(dim=-1, keepdim=True)
            var = gcn_out.var(dim=-1, keepdim=True, unbiased=False)
            print(var.min(), mean.min(), var.max(), mean.max(), gcn_out.shape)
            print("NaNs found in GCN output after LayerNorm:", gcn_out)
            raise RuntimeError("NaNs found in GCN output after LayerNorm")
        if torch.isinf(gcn_out).any():
            print("in", gcn_in.shape, gcn_in.min(), gcn_in.max())
            mean = gcn_out.mean(dim=-1, keepdim=True)
            var = gcn_out.var(dim=-1, keepdim=True, unbiased=False)
            print(var.min(), mean.min(), var.max(), mean.max(), gcn_out.shape)
            print("Inf found in GCN output after LayerNorm:", gcn_out)
            raise RuntimeError("NaNs found in GCN output after LayerNorm")

        gcn_out = self.ln_gcn(gcn_out)
        gcn_out = self.dropout(gcn_out)

        # Classifier
        logits = self.classifier(gcn_out)
        # print(logits)
        if torch.isnan(logits).any():
            print("NaNs found in GCN output after logits:", logits)
            raise RuntimeError("NaNs found in GCN in logits")
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # Assuming labels are flattened to (total_num_words,)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(logits.view(-1, self.num_labels), labels.view(-1))
                print(loss)
                raise RuntimeError("NaNs or Infs found in Loss")

        if not return_dict:
             raise Exception("return_dict=False is not fully implemented in the restructured code.")

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BERTWithGCNForTokenClassificationResidual(BertPreTrainedModel):
    config_class = BertConfig

    def __init__(self, config, pos_vocab_size=None, dep_vocab_size=None, merge_strategy='average'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = AutoModel.from_config(config)
        self.relu = torch.nn.ReLU() # Added activation as discussed previously
        self.gelu = torch.nn.GELU()

        self.pos_emb = torch.nn.Embedding(pos_vocab_size, int(config.hidden_size * 0.1))
        self.dep_emb = torch.nn.Embedding(dep_vocab_size, int(config.hidden_size * 0.1))
        self.ln_pre_gcn = torch.nn.LayerNorm(config.hidden_size + 2 * int(config.hidden_size * 0.1), eps=1e-5)
        self.reduction_layer = torch.nn.Linear(config.hidden_size + 2 * int(config.hidden_size * 0.1),
                                               128)
        self.ln_reduction = torch.nn.LayerNorm(128, eps=1e-5)

        self.cat_norm = torch.nn.LayerNorm(128, eps=1e-5)
        # self.ln_word_emb = torch.nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ln_gcn = torch.nn.LayerNorm(128, eps=1e-5)

        self.gcn = GCNConv(in_channels=128,
                           out_channels=128,
                           add_self_loops=True
                           )
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(128, # + 2 * int(config.hidden_size * 0.1),
                                          config.num_labels)

        self.post_init()


        torch.nn.init.xavier_uniform_(self.gcn.lin.weight)
        torch.nn.init.zeros_(self.gcn.bias)

        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.zeros_(self.classifier.bias)

        torch.nn.init.xavier_uniform_(self.reduction_layer.weight)
        torch.nn.init.zeros_(self.reduction_layer.bias)

        torch.nn.init.xavier_uniform_(self.pos_emb.weight, gain=0.01)
        torch.nn.init.xavier_uniform_(self.dep_emb.weight, gain=0.01)

        with torch.no_grad():
            self.gcn.lin.weight.data.mul_(0.01)
            self.classifier.weight.data.mul_(0.01)
            self.reduction_layer.weight.data.mul_(0.01)
            self.pos_emb.weight.data.mul_(0.01)
            self.dep_emb.weight.data.mul_(0.01)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            pos_ids: Optional[torch.Tensor] = None,
            dep_ids: Optional[torch.Tensor] = None,
            edge_index: Optional[List[torch.Tensor]] = None,
            word_ids: Optional[list] = None, # list of lists
            labels: Optional[torch.Tensor] = None, # flattened labels (total_words)
            word_count: Optional[List[int]] = None, # list of num_words per sentence
            # ----------------------------------------------
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if word_ids is None or word_count is None or edge_index is None:
             raise ValueError("word_ids, word_count, and edge_index are required inputs.")

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]

        pos_embed = self.pos_emb(pos_ids)  # Shape: (batch_size, max_seq_length, 50)
        dep_embed = self.dep_emb(dep_ids)  # Shape: (batch_size, max_seq_length, 50)

        # Combine BERT output with POS and DEP embeddings
        sequence_output = torch.cat([sequence_output, pos_embed, dep_embed], dim=-1)
        sequence_output = self.ln_pre_gcn(sequence_output)
        sequence_output = self.gelu(sequence_output)
        sequence_output = self.ln_reduction(self.gelu(self.reduction_layer(sequence_output)))

        if torch.isnan(sequence_output).any():
            print(sequence_output)
            raise RuntimeError("NaNs after bert layer")

        batch_size, seq_length, hidden_size = sequence_output.shape
        batch_word_embeddings = []
        for i in range(batch_size):
            # Use a set to track word IDs seen *for the first time* in this sentence
            seen_word_ids_in_sentence = set()
            word_embeddings_list = []
            for j in range(word_count[i]):
                if j in word_ids[i]:
                    word_embeddings_list.append(sequence_output[i][word_ids[i].index(j)])
                else:
                    word_embeddings_list.append(sequence_output[i][0])

            batch_word_embeddings.append(torch.stack(word_embeddings_list))


        # Create pytorch_geometric Data objects for the batch
        offset = 0
        edge_indices = [[], []]
        for i in range(batch_size):
            edge_index_sent, _ = add_self_loops(edge_index[i], num_nodes=batch_word_embeddings[i].size(0))
            edge_indices[0].extend(edge_index_sent[0] + offset)
            edge_indices[1].extend(edge_index_sent[1] + offset)
            offset += batch_word_embeddings[i].size(0)

        gcn_in = torch.cat(batch_word_embeddings, dim=0)
        edge_index = torch.tensor(edge_indices).to(self.device)
        deg = degree(edge_index[1], num_nodes=gcn_in.size(0)).clamp(min=1)
        edge_weight = 1.0 / deg[edge_index[1]]
        gcn_in = self.cat_norm(gcn_in)
        if torch.isnan(gcn_in).any():
            print(gcn_in)
            print("NaNs found in word embeddings after scatter/LayerNorm:", gcn_in)
            raise RuntimeError("NaNs found in word embeddings after scatter/LayerNorm")

        # print(edge_index.shape, edge_index.min(), edge_index.max())
        # print(edge_index)
        # print(gcn_out.shape)
        # GCN forward pass
        gcn_out = self.gcn(gcn_in, edge_index, edge_weight=edge_weight.abs())
        gcn_out = gcn_out + gcn_in


        if torch.isnan(gcn_out).any():
            mean = gcn_out.mean(dim=-1, keepdim=True)
            var = gcn_out.var(dim=-1, keepdim=True, unbiased=False)
            print(var.min(), mean.min(), var.max(), mean.max(), gcn_out.shape)
            print("NaNs found in GCN output after LayerNorm:", gcn_out)
            raise RuntimeError("NaNs found in GCN output after LayerNorm")
        if torch.isinf(gcn_out).any():
            mean = gcn_out.mean(dim=-1, keepdim=True)
            var = gcn_out.var(dim=-1, keepdim=True, unbiased=False)
            print(var.min(), mean.min(), var.max(), mean.max(), gcn_out.shape)
            print("Inf found in GCN output after LayerNorm:", gcn_out)
            raise RuntimeError("NaNs found in GCN output after LayerNorm")

        gcn_out = self.gelu(gcn_out)
        gcn_out = self.ln_gcn(gcn_out)
        gcn_out = self.dropout(gcn_out)

        # Classifier
        logits = self.classifier(gcn_out)
        # print(logits)
        if torch.isnan(logits).any():
            print("NaNs found in GCN output after LayerNorm:", logits)
            raise RuntimeError("NaNs found in GCN in logits")
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # Assuming labels are flattened to (total_num_words,)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(logits.view(-1, self.num_labels), labels.view(-1))
                print(loss)
                raise RuntimeError("NaNs or Infs found in Loss")

        if not return_dict:
             raise Exception("return_dict=False is not fully implemented in the restructured code.")

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


