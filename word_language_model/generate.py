
                
###############################################################################
# Language Modeling on Custom Dataset (Parliament Subset)
#
# This file generates new sentences sampled from the language model.
###############################################################################

import argparse
import torch
import model as model_file
from torch.serialization import safe_globals

#新增
import pickle

parser = argparse.ArgumentParser(description='PyTorch Custom Language Model')
# Model parameters
parser.add_argument('--data', type=str, default='./data/parliament_subset',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')

#以下两行为新增vocab参数
parser.add_argument('--vocab', type=str, default='./vocab.pkl',
                    help='path to vocabulary file')

parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default=1000,
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                    help='enables macOS GPU training')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set device
torch.manual_seed(args.seed)
if torch.cuda.is_available() and args.cuda:
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and args.mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Validate temperature
if args.temperature < 1e-3:
    parser.error("--temperature has to be >= 1e-3")

# Load model (with safe class context)
with safe_globals({'RNNModel': model_file.RNNModel}):
    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device, weights_only=False)

model.eval()


# 新增Load vocabulary for index-to-word conversion
with open(args.vocab, 'rb') as f:
    idx2word = pickle.load(f)
    
    
# Use vocab size from the model itself
ntokens = model.encoder.num_embeddings
is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'

# Initialize hidden state & input
if not is_transformer_model:
    hidden = model.init_hidden(1)

input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

# Generate text
with open(args.outf, 'w', encoding='utf-8') as outf:
    with torch.no_grad():
        for i in range(args.words):
            if is_transformer_model:
                output = model(input, False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
            else:
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)

            #word = str(word_idx.item())  # 输出为 token ID，可替换为词典映射
            #word = idx2word.get(word_idx.item(), "<unk>")  # 反查词表
            #s上面这一行保存的词表 idx2word 是一个 list 类型，而不是 dict，所以不能用 .get() 方法。
            word = idx2word[word_idx.item()] if word_idx.item() < len(idx2word) else "<unk>"

            outf.write(word + ('\n' if i % 20 == 19 else ' '))
            if i % args.log_interval == 0:
                print(f'| Generated {i}/{args.words} words')
                
