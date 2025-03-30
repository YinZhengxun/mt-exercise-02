# Word-level Language Modeling using RNN and Transformer

This example trains a multi-layer RNN (Elman, GRU, or LSTM) or Transformer on a language modeling task. By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash
python main.py --cuda --epochs 6           # Train a LSTM on Wikitext-2 with CUDA.
python main.py --cuda --epochs 6 --tied    # Train a tied LSTM on Wikitext-2 with CUDA.
python main.py --cuda --tied               # Train a tied LSTM on Wikitext-2 with CUDA for 40 epochs.
python main.py --cuda --epochs 6 --model Transformer --lr 5
                                           # Train a Transformer model on Wikitext-2 with CUDA.

python generate.py                         # Generate samples from the default model checkpoint.
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`) or Transformer module (`nn.TransformerEncoder` and `nn.TransformerEncoderLayer`) which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received, training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --mps                 enable GPU on macOS
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --nhead NHEAD         the number of heads in the encoder/decoder of the transformer model
  --dry-run             verify the code and the model
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied
```

###DATASET  EXPERIMENT DESCRIPTION 
'''
I. Dataset Description
For this experiment, we used the Parliament Corpus from the Cornell ConvoKit, which contains real transcripts from British parliamentary debates. The language is formal and grammatically rich, making it a suitable dataset for training a language model.

The original dataset is in .jsonl format. We wrote a preprocessing script extract_parliament.py to extract the "utterance" text field and convert it to plain text.

To balance model quality and training efficiency, we extracted approximately 10,000 text snippets, which were randomly shuffled and split into train.txt, valid.txt, and test.txt stored in the data/parliament_subset/ directory.


II. Code Modifications
Preprocessing Script

Added extract_parliament.py to parse and prepare the text data from utterances.jsonl.

Training Script (main.py)

Modified to accept the custom dataset path (--data data/parliament_subset).

During training, the vocabulary (idx2word) is saved as a vocab.pkl file for use during generation.

Text Generation Script (generate.py)

Added --vocab argument to load the vocabulary file.

The generated output now uses actual words instead of token IDs, using reverse lookup from the vocabulary.

III. Generation Output
The generated text reflects the formal tone and structure of parliamentary speech. Sample output includes proper use of sentence boundaries, rhetorical phrasing, and named entities like:

"I thank my hon. Friend for his question. It is important that we continue to support local authorities..."

While some sequences are syntactically awkward, the model demonstrates promising coherence given limited data and training epochs.

