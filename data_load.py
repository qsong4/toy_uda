import tensorflow as tf
from utils import calc_num_batches
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as kr
from gensim.models import Word2Vec
import random


# rng = random.Random(5)

def loadGloVe(filename):
    embd = np.load(filename)
    return embd


def loadGloVe_2(filename, emb_size):
    mu, sigma = 0, 0.1  # 均值与标准差
    rarray = np.random.normal(mu, sigma, emb_size)
    rarray_cls = np.random.normal(mu, sigma, emb_size)
    embd = {}
    embd['<pad>'] = [0] * emb_size
    # embd['<pad>'] = list(rarray)
    embd['<unk>'] = list(rarray)
    # embd['<cls>'] = list(rarray_cls)
    file = open(filename, 'r')
    for line in tqdm(file.readlines()):
        row = line.rstrip().split(' ')
        if row[0] in embd.keys():
            continue
        else:
            embd[row[0]] = [float(v) for v in row[1:]]
    file.close()
    return embd


def loadw2v(filename, emb_size):
    mu, sigma = 0, 0.1  # 均值与标准差
    rarray = np.random.normal(mu, sigma, emb_size)
    embd = {}
    embd['<pad>'] = [0] * emb_size
    # embd['<pad>'] = list(rarray)
    embd['<unk>'] = list(rarray)
    w2v = Word2Vec.load(filename)
    for word in w2v.wv.vocab:
        if word in embd.keys():
            continue
        else:
            embd[word] = w2v[word]
    return embd


def preprocessVec(gloveFile, vocab_file, outfile):
    emdb = loadGloVe_2(gloveFile, 300)
    # emdb = loadw2v(gloveFile, 300) #中文w2v用的
    trimmd_embd = []
    with open(vocab_file, 'r') as fr:
        for line in fr:
            word = line.rstrip()
            if word in emdb:
                trimmd_embd.append(emdb[word])
            else:
                trimmd_embd.append(emdb['<unk>'])
    np.save(outfile, trimmd_embd)


def load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <mask>, 3: <cls>

    Returns
    two dictionaries.
    '''
    with open(vocab_fpath, 'r') as fr:
        vocab = [line.strip() for line in fr]

    # vocab = [line.split()[0] for line in open(vocab_fpath, 'r').read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token, vocab


def load_data(fpath, maxlen):
    '''Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.

    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    sents1, sents2 = [], []
    labels = []
    with open(fpath, 'r') as fr:
        for line in fr:
            content = line.strip().split("\t")
            sent1 = content[0].lower()
            sent2 = content[1].lower()
            # label = int(content[2]) #cn data
            label = content[2]  # snli data
            if len(sent1.split()) > maxlen:
                continue
                # sent1 = sent1[len(sent1) - maxlen:]#for cn data
            if len(sent2.split()) > maxlen:
                continue
                # sent2 = sent2[len(sent2) - maxlen:]#for cn data
            sents1.append(sent1)
            sents2.append(sent2)
            labels.append([label])
    return sents1, sents2, labels


def removePunc(inputStr):
    string = re.sub(r"\W+", "", inputStr)
    return string.strip()


def encode(inp, dict, maxlen):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    '''
    # for cn dataset
    # x = [dict.get(t, dict["<unk>"]) for t in inp]
    # for snli dateset
    x = []
    # x.append(dict["<cls>"])#第一个是cls标签
    for i in re.split(r"\W+", inp):
        i = i.strip()
        i = removePunc(i)
        i = i.lower()
        if i == "":
            continue
        x.append(dict.get(i, dict["<unk>"]))
    x_len = len(x)
    x = pad_sequences([x], maxlen=maxlen, dtype='int32', padding='post')
    # x = [dict.get(t, dict["<unk>"]) for t in re.split(r"\W+'", inp)]
    return x[0], x_len

# process for snli task
def process_file_sup(fpath, vocab_fpath, maxlen, rng):
    token2idx, _, vocab = load_vocab(vocab_fpath)
    vocab_len = len(vocab)
    with open(fpath, 'r') as fr:

        sentences = list(fr.readlines())
        rng.shuffle(sentences)
        sent_len = len(sentences)

        inputs_a = np.zeros((sent_len, maxlen))
        a_lens = np.zeros(sent_len)

        related_labels = []

        for index, sent in tqdm(enumerate(sentences)):
            content = sent.strip().split('\t')
            senta = content[0]
            real_label = content[2]

            enc_a, len_a = encode(senta, token2idx, maxlen)

            inputs_a[index, :] = np.array(enc_a)
            a_lens[index] = len_a
            related_labels.append([real_label])

    # 这里是二分类任务，所以numcls设置为了2
    label_enc = OneHotEncoder(sparse=False, categories='auto')
    related_labels = label_enc.fit_transform(related_labels)
    # related_labels = kr.utils.to_categorical(related_labels, num_classes = 2)

    print("***********data example***********")
    print("enc_a: ", inputs_a[0, :])
    print("related_labels: ", related_labels[0, :])

    return (inputs_a, a_lens, related_labels)

# process for snli task
def process_file_unsup(fpath, vocab_fpath, maxlen, maxsentences=20000):
    token2idx, _, vocab = load_vocab(vocab_fpath)
    total_text = {}
    ori_input = []
    aug_input = []
    with open(fpath, 'r') as fr:
        print("process unsup stage1: ")
        for index, sent in tqdm(fr):
            content = sent.strip().split('\t')
            senta = content[0]
            real_label = content[2]
            if real_label in total_text:
                total_text[real_label].append(senta)
            else:
                total_text[real_label] = [senta]
        print("process unsup stage2: ")
        for k, v in tqdm(total_text.items()):
            len_v = len(v)
            if len_v>maxsentences:
                v = v[:maxsentences]
            len_v = len(v)
            if len_v%2 == 0:
                ori_input += v[:len_v/2]
                aug_input += v[len_v/2:]
            else:
                len_v = len_v-1
                ori_input += v[:len_v/2]
                aug_input += v[len_v/2:]

            assert len(ori_input) == len(aug_input)

        inputs_ori = np.zeros((len(ori_input), maxlen))
        inputs_aug = np.zeros((len(ori_input), maxlen))
        ori_lens = np.zeros(len(ori_input))
        aug_lens = np.zeros(len(ori_input))

        for index, x in tqdm(enumerate(ori_input)):
            enc_a, len_a = encode(x, token2idx, maxlen)
            inputs_ori[index, :] = np.array(enc_a)
            ori_lens[index] = len_a

        for index, x in tqdm(enumerate(aug_input)):
            enc_a, len_a = encode(x, token2idx, maxlen)
            inputs_aug[index, :] = np.array(enc_a)
            aug_lens[index] = len_a

    print("***********data example***********")
    print("ori: ", ori_input[0, :])
    print("ori len: ", ori_lens[0, :])
    print("aug: ", aug_input[0, :])
    print("aug len: ", aug_lens[0, :])

    return (ori_input, ori_lens, aug_input, aug_lens)

def get_batch_unsup(features, batch_size, shuffle=True):
    ori_input, ori_lens, aug_input, aug_lens = features

    instance_len = len(ori_input)
    num_batches = calc_num_batches(instance_len, batch_size)

    if shuffle:
        indices = np.random.permutation(np.arange(instance_len))
        ori_input = ori_input[indices]
        ori_lens = ori_lens[indices]
        aug_input = aug_input[indices]
        aug_lens = aug_lens[indices]

    for i in range(num_batches):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, instance_len)
        yield (ori_input[start_id:end_id], ori_lens[start_id:end_id],
               aug_input[start_id:end_id], aug_lens[start_id:end_id])

def get_batch_sup(features, batch_size, shuffle=True):

    inputs_a, a_lens, related_labels = features

    instance_len = len(inputs_a)
    num_batches = calc_num_batches(instance_len, batch_size)

    if shuffle:
        indices = np.random.permutation(np.arange(instance_len))
        inputs_a = inputs_a[indices]
        a_lens = a_lens[indices]
        related_labels = related_labels[indices]

    for i in range(num_batches):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, instance_len)
        yield (inputs_a[start_id:end_id], a_lens[start_id:end_id], related_labels[start_id:end_id])


if __name__ == '__main__':
    preprocessVec("./data/vec/glove.840B.300d.txt", "./data/snli.vocab", "./data/vec/snli_trimmed_vec.npy")
