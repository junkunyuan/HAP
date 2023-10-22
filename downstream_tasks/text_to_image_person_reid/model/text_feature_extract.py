from torch import nn
import torch
import transformers as ppb

class TextExtract(nn.Module):

    def __init__(self, opt):
        super(TextExtract, self).__init__()

        self.opt = opt
        self.last_lstm = opt.last_lstm
        self.embedding = nn.Embedding(opt.vocab_size, 512, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(512, 384, num_layers=1, bidirectional=True, bias=False)

    def forward(self, caption_id, text_length):

        text_embedding = self.embedding(caption_id)
        text_embedding = self.dropout(text_embedding)
        feature = self.calculate_different_length_lstm(text_embedding, text_length, self.lstm)

        return feature

    def calculate_different_length_lstm(self, text_embedding, text_length, lstm):

        text_length = text_length.view(-1)
        _, sort_index = torch.sort(text_length, dim=0, descending=True)
        _, unsort_index = sort_index.sort()

        sortlength_text_embedding = text_embedding[sort_index, :]
        sort_text_length = text_length[sort_index]
        packed_text_embedding = nn.utils.rnn.pack_padded_sequence(sortlength_text_embedding,
                                                                  sort_text_length,
                                                                  batch_first=True)

        packed_feature, [hn, _] = lstm(packed_text_embedding)  # [hn, cn]
        sort_feature = nn.utils.rnn.pad_packed_sequence(packed_feature, batch_first=True)  # including[feature, length]
        # print(hn.size(), cn.size())

        if self.last_lstm:
            hn = torch.cat([hn[0, :, :], hn[1, :, :]], dim=1)[unsort_index, :]
            return hn
        else:
            unsort_feature = sort_feature[0][unsort_index, :]
            unsort_feature = (unsort_feature[:, :, :int(unsort_feature.size(2) / 2)]
                              + unsort_feature[:, :, int(unsort_feature.size(2) / 2):]) / 2
            return unsort_feature

class TextExtract_Bert_lstm(nn.Module):
    def __init__(self, args):
        super(TextExtract_Bert_lstm, self).__init__()

        self.last_lstm = args.last_lstm
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert_base_uncased')
        self.text_embed = model_class.from_pretrained(pretrained_weights)
        self.text_embed.eval()
        for p in self.text_embed.parameters():
            p.requires_grad = False
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(768, 768, num_layers=1, bidirectional=True, bias=False)

    def forward(self, txt, mask):
        length = mask.sum(1)
        length = length.cpu()
        with torch.no_grad():
            txt = self.text_embed(txt, attention_mask=mask)#
            txt = txt[0]   ##64 * L * 768

        txt = self.calculate_different_length_lstm(txt,length,self.lstm)
        return txt

    def calculate_different_length_lstm(self, text_embedding, text_length, lstm):

        text_length = text_length.view(-1)
        _, sort_index = torch.sort(text_length, dim=0, descending=True)
        _, unsort_index = sort_index.sort()

        sortlength_text_embedding = text_embedding[sort_index, :]
        sort_text_length = text_length[sort_index]
        # print(sort_text_length)
        packed_text_embedding = nn.utils.rnn.pack_padded_sequence(sortlength_text_embedding,
                                                                  sort_text_length,
                                                                  batch_first=True)

        packed_feature, [hn, _] = lstm(packed_text_embedding)  # [hn, cn]
        sort_feature = nn.utils.rnn.pad_packed_sequence(packed_feature, batch_first=True)  # including[feature, length]
        # print(hn.size(), cn.size())

        if self.last_lstm:
            hn = torch.cat([hn[0, :, :], hn[1, :, :]], dim=1)[unsort_index, :]
            return hn
        else:
            unsort_feature = sort_feature[0][unsort_index, :]
            unsort_feature = (unsort_feature[:, :, :int(unsort_feature.size(2) / 2)]
                              + unsort_feature[:, :, int(unsort_feature.size(2) / 2):]) / 2
            return unsort_feature