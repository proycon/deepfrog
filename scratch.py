import sys
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

batch_size = 32
if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
else:
    print("WARNING: No GPU found, running on CPU",file=sys.stderr)
    device = torch.device("cpu")


class POSData:

    def __init__(self, tokeniser, maxtokens=50):
        self.tags = set() #list of all tags
        self.index2tag = []
        self.tag2index = {}
        self.maxtokens = maxtokens #maximum number of tokens per sentence
        self.maxlength = 0 #maximum number of subtokens per sentence (computed later)
        self.sentences = []
        self.tokeniser = tokeniser

    def load_mbt_file(self,filename):
        """Load an MBT-style training file, returns tagged sentences"""
        sentence = []
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                fields = line.strip().split("\t")
                if fields:
                    if fields[0] == "<utt>":
                        #end of sentence marker-found
                        if len(sentence) <= self.maxtokens:
                            self.sentences.append(sentence)
                        else:
                            print("Skipping sentence " + str(i+1) + " because it exceeds the maximum token limit",file=sys.stderr)
                        sentence = []
                    else:
                        word, tag = fields
                        sentence.append(word, tag)
                        self.tags.add(tag)
        if sentence and len(sentence) <= self.maxtokens: #in case the final <utt> is omitted
            self.sentences.append(sentence)

    def build_index(self):
        """Build the index (only needs to be done once and is automatically invoked when needed)"""
        self.index2tag = ["[PAD]"] + list(sorted(self.tags))
        self.tag2index = { tag:i for i, tag in enumerate(self.index2tag)}

    def __getitem__(self, i):
        """Returned a tokenised/indexed version of the sentence at the specified index, suitable for plugging into torch.utils.data.DataLoader"""
        if not self.tag2index:
            self.build_index()

        words = ["[CLS]"] + [ x[0] for x in self.sentences[i]] + ["[SEP]"]
        tags = ["[PAD]"] + [ x[1] for x in self.sentences[i]] + ["[PAD]"]

        tokenids = []
        tagids = []
        head_mask = [] #list of boolean ints, flags if the token is the first piece of a word
        for word, tag in zip(words, tags):
            #a word can expand to multiple (sub)tokens for BERT et al
            #(here we basically do manually what tokeniser.encode / encode_plus does automatically, but I'd rather be explicit and have control and insight)
            wordtokens = self.tokeniser.tokenize(word) if word not in ("[CLS]","[SEP]") else [word]
            wordtokenids = self.tokeniser.convert_tokens_to_ids(wordtokens)

            #we only set a tag for the first token of the word, the rest is set to [PAD] to represent there is no decision there
            wordtags =  [tag]
            wordtags += ["[PAD]"] * (len(wordtokens) - 1)
            wordtagids = [ self.tag2index[tag2] for tag2 in wordtags ]

            #flag the beginning of the word
            wordhead_mask = [1]
            wordhead_mask += [0] * (len(wordtokens) - 1)

            head_mask += wordhead_mask
            tokenids += wordtokenids
            tagids += wordtagids

        assert len(tokenids) == len(tagids) == len(head_mask)

        l = len(tokenids)
        if l > self.maxlength:
            self.maxlength = l

        return " ".join(words), tokenids, head_mask, " ".join(tags), tagids, l

    def __len__(self):
        return len(self.sentences)

def pad(batch):
    """Pads to the longest sample"""

    #index number refers to indices respective to POSData.__getitem__

    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    head_mask = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0 == [PAD]
    tokenids = f(1, maxlen)
    tagids = f(-2, maxlen)

    return words, torch.LongTensor(tokenids), head_mask, tags, torch.LongTensor(tagids), seqlens




def main():
    # Load pre-trained model tokenizer (vocabulary)
    tokeniser = BertTokenizer.from_pretrained("bert-base-dutch-cased")

    trainfile = sys.argv[1]

    traindata = POSData(tokeniser)
    traindata.load_mbt_file(trainfile)

    #testdata = POSData(tokeniser)
    #testdata.load_mbt_file(testfile)

    train_iter = torch.utils.data.DataLoader(data=traindata, batch_size=8, shuffle=True, num_workers=1, collate_fn=pad)


    model = BertForTokenClassification.from_pretrained("bert-base-dutch-cased")
    outputs = model(

MAX_SENTENCE_LEN = 75













