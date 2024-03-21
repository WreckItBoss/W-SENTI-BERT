import torch.nn as nn
import torch
import os


class BERTSeqClf(nn.Module):
    def __init__(self, num_labels, model='bert-base', n_layers_freeze=0, wiki_model='', n_layers_freeze_wiki=0, n_layers_freeze_senti=0):
        super(BERTSeqClf, self).__init__()

        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from transformers import AutoModel
        if model == 'bert-base':
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
        elif model == 'bertweet':
            self.bert = AutoModel.from_pretrained('vinai/bertweet-base')
        else:  # covid-twitter-bert
            self.bert = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
     

        n_layers = 12 if model != 'covid-twitter-bert' else 24
        if n_layers_freeze > 0:
            n_layers_ft = n_layers - n_layers_freeze
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.bert.pooler.parameters():
               param.requires_grad = True
            for i in range(n_layers - 1, n_layers - 1 - n_layers_ft, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        if wiki_model:
            if wiki_model == model:
                self.bert_wiki = self.bert
                #SENTI FOR VAST
                self.bert_senti = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
            else:  # bert-base
                self.bert_wiki = AutoModel.from_pretrained('bert-base-uncased')
                #sentiBERT
                self.bert_senti = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")


            n_layers = 12
            if n_layers_freeze_wiki > 0:
                print("ALL RETARDS RISE UP")
                n_layers_ft = n_layers - n_layers_freeze_wiki
                for param in self.bert_wiki.parameters():
                    param.requires_grad = False
                for param in self.bert_wiki.pooler.parameters():
                    param.requires_grad = True
                for i in range(n_layers - 1, n_layers - 1 - n_layers_ft, -1):
                    for param in self.bert_wiki.encoder.layer[i].parameters():
                        param.requires_grad = True
            
            #SentiBERT Freeze Layer 
            n_layers_freeze_senti=11 #SentiBERTの固定する層の数（もしも上の1層だけを学習したければここは11.　2層の場合 10)
            n_layers_ft_senti = n_layers - n_layers_freeze_senti
            for param in self.bert_senti.parameters():
               param.requires_grad = False
            for param in self.bert_senti.pooler.parameters():
                    param.requires_grad = True
            for i in range(n_layers - 1, n_layers - 1 - n_layers_ft_senti, -1):
                    for param in self.bert_senti.encoder.layer[i].parameters():
                        param.requires_grad = True

        self.sigmoid = nn.Sigmoid()
        config = self.bert.config
           

        if wiki_model and wiki_model != model:
            #hidden = config.hidden_size + self.bert_wiki.config.hidden_size
            #senti below
            self.dense = nn.Linear(self.bert_wiki.config.hidden_size+self.bert_senti.config.hidden_size, config.hidden_size)
            self.gate_dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
            self.fc = nn.Linear(768*2,1)
            hidden = config.hidden_size
            nn.init.xavier_uniform_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
        else:#VAST
            hidden = config.hidden_size
            #senti below
            self.gate_dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
            nn.init.xavier_uniform_(self.gate_dense.weight)
            nn.init.zeros_(self.gate_dense.bias)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden, num_labels)
        self.model = model

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                input_ids_wiki=None, attention_mask_wiki=None, input_ids_senti = None,attention_mask_senti = None ):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True)

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        if input_ids_wiki is not None:
            outputs_wiki = self.bert_wiki(input_ids_wiki, attention_mask=attention_mask_wiki)
            pooled_output_wiki = outputs_wiki.pooler_output
            pooled_output_wiki = self.dropout(pooled_output_wiki)
 
            #Senti things from here
            outputs_senti = self.bert_senti(input_ids_senti, attention_mask=attention_mask_senti)
            pooled_output_senti = outputs_senti.pooler_output
            pooled_output_senti = self.dropout(pooled_output_senti)
            #GATING
            concat_wiki_and_senti=torch.cat((pooled_output_wiki, pooled_output_senti), dim=1)
            wikisenti = self.dense(concat_wiki_and_senti)
            pooled_output_overall = torch.cat((pooled_output, wikisenti), dim = 1)
            gate = self.sigmoid(self.gate_dense(pooled_output_overall))
            output = (1 - gate) * pooled_output + gate * wikisenti

            
        else: #VAST
            outputs_senti = self.bert_senti(input_ids_senti, attention_mask=attention_mask_senti)
            pooled_output_senti = outputs_senti.pooler_output
            pooled_output_senti = self.dropout(pooled_output_senti)
            pooled_output_overall = torch.cat((pooled_output, pooled_output_senti), dim = 1)
            gate = self.sigmoid(self.gate_dense(pooled_output_overall))
            output = (1 - gate) * pooled_output + gate * pooled_output_senti

        logits = self.classifier(output)
        return logits
