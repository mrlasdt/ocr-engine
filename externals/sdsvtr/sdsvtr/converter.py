import torch
import numpy as np

class BaseConvertor:
    """Convert between text, index and tensor for text recognize pipeline.

    Args:
        dict_type (str): Type of dict, options are 'DICT36', 'DICT37', 'DICT90'
            and 'DICT91'.
        dict_file (None|str): Character dict file path. If not none,
            the dict_file is of higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, the list
            is of higher priority than dict_type, but lower than dict_file.
    """
    start_idx = end_idx = padding_idx = 0
    unknown_idx = None
    lower = False

    DICT36 = tuple('0123456789abcdefghijklmnopqrstuvwxyz')
    DICT63 = tuple('0123456789abcdefghijklmnopqrstuvwxyz'
                   'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    DICT90 = tuple('0123456789abcdefghijklmnopqrstuvwxyz'
                   'ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()'
                   '*+,-./:;<=>?@[\\]_`~')
    DICT131 = tuple('0123456789abcdefghijklmnopqrstuvwxyz'
                   '!"#$%&\'()'
                   '*+,-./:;<=>?@[\\]_`~'
                   'ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ')
    DICT224 = tuple('0123456789abcdefghijklmnopqrstuvwxyz'
                   'ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()'
                   '*+,-./:;<=>?@[\\]_`~{}|^ ̂'
                   'ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ'
                   'ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ✪')

    def __init__(self, dict_type='DICT90'):
        assert dict_type in ('DICT36', 'DICT63', 'DICT90', 'DICT131', 'DICT224')
        self.idx2char = []
        
        if dict_type == 'DICT36':
            self.idx2char = list(self.DICT36)
        elif dict_type == 'DICT63':
            self.idx2char = list(self.DICT63)
        elif dict_type == 'DICT90':
            self.idx2char = list(self.DICT90)
        elif dict_type == 'DICT131':
            self.idx2char = list(self.DICT131)
        elif dict_type == 'DICT224':
            self.idx2char = list(self.DICT224)
        else:
            raise ('Dictonary not implemented')

        assert len(set(self.idx2char)) == len(self.idx2char), \
            'Invalid dictionary: Has duplicated characters.'

        self.char2idx = {char: idx for idx, char in enumerate(self.idx2char)}

    def num_classes(self):
        """Number of output classes."""
        return len(self.idx2char)


class AttnConvertor(BaseConvertor):
    """Convert between text, index and tensor for encoder-decoder based
    pipeline.

    Args:
        dict_type (str): Type of dict, should be one of {'DICT36', 'DICT90'}.
        dict_file (None|str): Character dict file path. If not none,
            higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, higher
            priority than dict_type, but lower than dict_file.
        with_unknown (bool): If True, add `UKN` token to class.
        max_seq_len (int): Maximum sequence length of label.
        lower (bool): If True, convert original string to lower case.
        start_end_same (bool): Whether use the same index for
            start and end token or not. Default: True.
    """

    def __init__(self,
                 dict_type='DICT90',
                 with_unknown=True,
                 max_seq_len=40,
                 lower=False,
                 start_end_same=True,
                 return_confident=False,
                 **kwargs):
        super().__init__(dict_type)
        assert isinstance(with_unknown, bool)
        assert isinstance(max_seq_len, int)
        assert isinstance(lower, bool)

        self.with_unknown = with_unknown
        self.max_seq_len = max_seq_len
        self.lower = lower
        self.start_end_same = start_end_same
        self.return_confident = return_confident

        self.update_dict()

    def update_dict(self):
        start_end_token = '<BOS/EOS>'
        unknown_token = '<UKN>'
        padding_token = '<PAD>'

        # unknown
        self.unknown_idx = None
        if self.with_unknown:
            self.idx2char.append(unknown_token)
            self.unknown_idx = len(self.idx2char) - 1

        # BOS/EOS
        self.idx2char.append(start_end_token)
        self.start_idx = len(self.idx2char) - 1
        if not self.start_end_same:
            self.idx2char.append(start_end_token)
        self.end_idx = len(self.idx2char) - 1

        # padding
        self.idx2char.append(padding_token)
        self.padding_idx = len(self.idx2char) - 1

        # update char2idx
        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx
    
    def __call__(self, indexes):
        strings = []
        confidents = []
        if self.return_confident:
            b,sq,_ = indexes.shape        
            for idx in range(b):
                index = indexes[idx, :, :]
                chars = index.argmax(-1)
                confident = index.max(-1)
                i = 0
                while i < sq and chars[i] != self.end_idx and chars[i] != self.padding_idx: i += 1
                chars = chars[:i]
                confident = confident[:i].min()
                string = [self.idx2char[i] for i in chars]
                strings.append(''.join(string))
                confidents.append(confident)
            
            return strings, confidents
        else:
            for index in indexes:
                i, l = 0, len(index)
                while i < l and index[i] != self.end_idx and index[i] != self.padding_idx: i += 1
                index = index[:i]
                string = [self.idx2char[i] for i in index]
                strings.append(''.join(string))
            return strings