import re
import os
import torch
import numpy as np
from torchtext.legacy.data import Pipeline, Dataset, Field, Iterator, Example, RawField, get_tokenizer
from torchtext.legacy.vocab import Vocab
from torchtext.data.utils import is_tokenizer_serializable, dtype_to_attr
from itertools import chain
from collections import Counter,OrderedDict

AMINO_MAP = {'<pad>':24, '*': 23, 'A': 0, 'C': 4, 'B': 20,
             'E': 6, 'D': 3, 'G': 7, 'F': 13, 'I': 9, 'H': 8,
             'K': 11, 'M': 12, 'L': 10, 'N': 2, 'Q': 5, 'P': 14,
             'S': 15, 'R': 1, 'T': 16, 'W': 17, 'V': 19, 'Y': 18,
             'X': 22, 'Z': 21}
             
             
             


             
AMINO_MAP_REV = ['A','R','N','D','C','Q','E','G','H','I','L','K',
                 'M','F','P','S','T','W','Y','V','B','Z','X','*','@']

AMINO_MAP_REV_ = ['A','R','N','D','C','Q','E','G','H','I','L','K',
                 'M','F','P','S','T','W','Y','V','N','Q','*','*','@']

def define_dataloader(X_pep, X_tcr, y=None,
                    maxlen_pep=None, maxlen_tcr=None, 
                    padding='mid',
                    batch_size=50, device='cuda'):

    device0 = 0 if device == 'cuda' else -1
    
    if maxlen_pep is None: maxlen_pep=max([len(x) for x in X_pep])
    if maxlen_tcr is None: maxlen_tcr=max([len(x) for x in X_tcr])

    # Define Field
    field_pep = Field_modified(tokenize=tokenizer, batch_first=True, 
                            pad_type=padding, fix_length=maxlen_pep)                       
    field_tcr = Field_modified(tokenize=tokenizer, batch_first=True, 
                            pad_type=padding, fix_length=maxlen_tcr)
    field_y = Field(sequential=False, use_vocab=False, dtype=torch.float32)

    # Define vocab
    amino_map = AMINO_MAP
    amino_map_rev = AMINO_MAP_REV
    
    field_pep.build_vocab()
    field_tcr.build_vocab()
    field_y.build_vocab() if y is not None else None

    field_pep.vocab.stoi = amino_map
    field_tcr.vocab.stoi = amino_map
    field_pep.vocab.itos = amino_map_rev
    field_tcr.vocab.itos = amino_map_rev
        
    # Define dataloader
    if y is None:
        fields = [('X_pep',field_pep), ('X_tcr',field_tcr), ('y',field_y)]
        example = [Example.fromlist([x1,x2,1.0], fields) for x1,x2 in zip(X_pep,X_tcr)]
    else:
        fields = [('X_pep',field_pep), ('X_tcr',field_tcr), ('y',field_y)]
        example = [Example.fromlist([x1,x2,x3], fields) for x1,x2,x3 in zip(X_pep,X_tcr,y)]
   
   ############
    if len(example) % batch_size == 1:
        example = example[:-1]

    dataset = Dataset(example, fields)
    loader = Iterator(dataset, batch_size=batch_size, device=device0, repeat=False, shuffle=True)

    data_loader = dict()
    data_loader['pep_amino_idx'] = field_pep.vocab.itos
    data_loader['tcr_amino_idx'] = field_tcr.vocab.itos
    data_loader['tensor_type'] = torch.cuda.LongTensor if device == "cuda"  else torch.LongTensor
    data_loader['pep_length'] = maxlen_pep
    data_loader['tcr_length'] = maxlen_tcr 
    data_loader['loader'] = loader

    return data_loader


def tokenizer(sequence):

    sequence = re.sub(r'\s+', '', str(sequence))
    sequence = re.sub(r'[^ARNDCQEGHILKMFPSTWYVBZX]', '*', sequence)
    sequence = [x for x in sequence]

    return sequence

def load_embedding(filename):
    '''
    read in BLOSUM matrix

    parameters:
        - filename : file containing BLOSUM matrix

    returns:
        - blosum embedding matrix: list
    '''
    if filename is None or filename.lower() == 'none':
        filename = 'data/blosum/BLOSUM45'
    
    embedding_file = open(filename, "r")
    lines = embedding_file.readlines()[7:]
    embedding_file.close()

    embedding = [[float(x) for x in l.strip().split()[1:]] for l in lines]
    embedding.append([0.0] * len(embedding[0]))
    
    return embedding

def load_data_split(x_pep, x_tcr, args):
    '''
    Split the data based on the method specified
    random - The data is split randomly into equal sized folds.
    peptide - The data is split such that no training peptides are present in the testing peptides
    tcr - The data is split such that no training tcrs are present in the testing tcrs.

    parameters:
        - x_pep
        - x_tcr
        - args

    returns:
        - idx_train - Indices for training data
        - idx_test - Indices for testing data
        - idx_test_remove - Indices for removed data (outer loop cross validation)
    '''

    split_type = args.split_type
    idx_test_remove = None
    idx_test = None
    idx_train = None

    if split_type == 'random':
        n_total = len(x_pep)
    elif split_type == 'epitope':
        unique_peptides = np.unique(x_pep)
        n_total = len(unique_peptides)
    elif split_type == 'tcr':
        unique_tcrs = np.unique(x_tcr)
        n_total = len(unique_tcrs)

    indexfile = re.sub('.csv', f'_{args.split_type}_data_shuffle.txt', args.infile)
    if os.path.exists(indexfile):
        idx_shuffled = np.loadtxt(indexfile, dtype=np.int32)
    else:
        idx_shuffled = np.arange(n_total)
        np.random.shuffle(idx_shuffled)
        np.savetxt(indexfile, idx_shuffled, fmt='%d')

    # Determine data split from folds
    n_test = int(round(n_total / args.n_fold))
    n_train = n_total - n_test


    # Determine position of current test fold
    test_fold_start_index = args.idx_test_fold * n_test
    test_fold_end_index = (args.idx_test_fold + 1) * n_test

    if split_type == 'random':
        # Split data evenly among evenly spaced folds
        # Determine if there is an outer testing fold
        if args.idx_val_fold < 0:
            idx_test = idx_shuffled[test_fold_start_index:test_fold_end_index]
            idx_train = list(set(idx_shuffled).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove = idx_shuffled[test_fold_start_index:test_fold_end_index]
            idx_test = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            idx_train = list(set(idx_shuffled).difference(set(idx_test)).difference(set(idx_test_remove)))
    elif split_type == 'epitope':
        if args.idx_val_fold < 0:
            idx_test_pep = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_peptides = unique_peptides[idx_test_pep]
            idx_test = [index for index, pep in enumerate(x_pep) if pep in test_peptides]
            idx_train = list(set(range(len(x_pep))).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove_pep = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_remove_peptides = unique_peptides[idx_test_remove_pep]
            idx_test_pep = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            test_peptides = unique_peptides[idx_test_pep]
            idx_test = [index for index, pep in enumerate(x_pep) if pep in test_peptides]
            idx_test_remove = [index for index, pep in enumerate(x_pep) if pep in test_remove_peptides]
            idx_train = list(set(range(len(x_pep))).difference(set(idx_test)).difference(set(idx_test_remove)))
    elif split_type == 'tcr':
        if args.idx_val_fold < 0:
            idx_test_tcr = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_tcrs = unique_tcrs[idx_test_tcr]
            idx_test = [index for index, tcr in enumerate(x_tcr) if tcr in test_tcrs]
            idx_train = list(set(range(len(x_tcr))).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove_tcr = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_remove_tcrs = unique_tcrs[idx_test_remove_tcr]
            idx_test_tcr = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            test_tcrs = unique_tcrs[idx_test_tcr]
            idx_test = [index for index, tcr in enumerate(x_tcr) if tcr in test_tcrs]
            idx_test_remove = [index for index, tcr in enumerate(x_tcr) if tcr in test_remove_tcrs]
            idx_train = list(set(range(len(x_tcr))).difference(set(idx_test)).difference(set(idx_test_remove)))

    return idx_train, idx_test, idx_test_remove

class Field_modified(RawField):
    """Modification of class Field
    Defines a datatype together with instructions for converting to Tensor.
    Field class models common text processing datatypes that can be represented
    by tensors.  It holds a Vocab object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The Field object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method and the kind of
    Tensor that should be produced.
    If a Field is shared between two columns in a dataset (e.g., question and
    answer in a QA dataset), then they will have a shared vocabulary.
    Attributes:
        sequential: Whether the datatype represents sequential data. If False,
            no tokenization is applied. Default: True.
        use_vocab: Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: True.
        init_token: A token that will be prepended to every example using this
            field, or None for no initial token. Default: None.
        eos_token: A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: None.
        fix_length: A fixed length that all examples using this field will be
            padded to, or None for flexible sequence lengths. Default: None.
        dtype: The torch.dtype class that represents a batch of examples
            of this kind of data. Default: torch.long.
        preprocessing: The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: None.
        postprocessing: A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list, and
            the field's Vocab.
            Default: None.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. If "spacy", the SpaCy tokenizer is
            used. If a non-serializable function is passed as an argument,
            the field will not be able to be serialized. Default: string.split.
        tokenizer_language: The language of the tokenizer to be constructed.
            Various languages currently supported only in SpaCy.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        batch_first: Whether to produce tensors with the batch dimension first.
            Default: False.
        pad_token: The string token used as padding. Default: "<pad>".
        unk_token: The string token used to represent OOV words. Default: "<unk>".
        pad_type: Padding type "front", "mid", "end" Default: "mid".
        truncate_first: Do the truncating of the sequence at the beginning. Default: False
        stop_words: Tokens to discard during the preprocessing step. Default: None
        is_target: Whether this field is a target variable.
            Affects iteration over batches. Default: False
    """

    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor dtypes to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }

    ignore = ['dtype', 'tokenize']

    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=None, tokenizer_language='en', include_lengths=False,
                 batch_first=False, pad_token="<pad>", unk_token="<unk>",
                 pad_type = 'mid', truncate_first=False, stop_words=None,
                 is_target=False):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        # store params to construct tokenizer for serialization
        # in case the tokenizer isn't picklable (e.g. spacy)
        self.tokenizer_args = (tokenize, tokenizer_language)
        self.tokenize = get_tokenizer(tokenize) #tokenizer_language
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token if self.sequential else None
        self.pad_type = pad_type
        self.truncate_first = truncate_first
        try:
            self.stop_words = set(stop_words) if stop_words is not None else None
        except TypeError:
            raise ValueError("Stop words must be convertible to a set")
        self.is_target = is_target

    def __getstate__(self):
        str_type = dtype_to_attr(self.dtype)
        if is_tokenizer_serializable(*self.tokenizer_args):
            tokenize = self.tokenize
        else:
            # signal to restore in `__setstate__`
            tokenize = None
        attrs = {k: v for k, v in self.__dict__.items() if k not in self.ignore}
        attrs['dtype'] = str_type
        attrs['tokenize'] = tokenize

        return attrs

    def __setstate__(self, state):
        state['dtype'] = getattr(torch, state['dtype'])
        if not state['tokenize']:
            state['tokenize'] = get_tokenizer(*state['tokenizer_args'])
        self.__dict__.update(state)

    def __hash__(self):
        # we don't expect this to be called often
        return 42

    def __eq__(self, other):
        if not isinstance(other, RawField):
            return False

        return self.__dict__ == other.__dict__

    
    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.
        If `sequential=True`, the input will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        if self.sequential and isinstance(x, str):
            x = self.tokenize(x.rstrip('\n'))
        if self.lower:
            x = Pipeline(str.lower)(x)
        if self.sequential and self.use_vocab and self.stop_words is not None:
            x = [w for w in x if w not in self.stop_words]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x
    

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.
        Pad, numericalize, and postprocess a batch and create a tensor.
        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
            and custom postprocessing Pipeline.
        """
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor
    
    def pad(self, minibatch):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.
        """
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_type=='front':
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x))
                    + ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token]))
            elif self.pad_type=='end':
                padded.append(
                    ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token])
                    + [self.pad_token] * max(0, max_len - len(x)))
            elif self.pad_type=='mid':
                i_gap = np.int32(np.ceil(min(len(x), max_len)/2))
                i_gap_rev = min(len(x), max_len) - i_gap
                padded.append(
                    ([] if self.init_token is None else [self.init_token])
                    + list(x[:i_gap])
                    + [self.pad_token] * max(0, max_len - len(x))
                    + list(x[-i_gap_rev:])
                    + ([] if self.eos_token is None else [self.eos_token]))
            else:
                raise ValueError('pad_type should be "front", "mid", or "end"')
         
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)

        return padded
   
    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.
        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token] + kwargs.pop('specials', [])
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)
        else:
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explicitly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, str)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        var = torch.tensor(arr, dtype=self.dtype, device=device)

        if self.sequential and not self.batch_first:
            var.t_()
        if self.sequential:
            var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var
