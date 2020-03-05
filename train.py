import tensorflow as tf

from model import toy_uda
from tqdm import tqdm
from data_load import process_file_sup, process_file_unsup, get_batch_sup, get_batch_unsup,calc_num_batches
from utils import save_variable_specs
import os
from hparams import Hparams
import pickle
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def evaluate(sess, dev_features):
    m.model = False
    #inputs_a, a_lens, related_labels
    dev_batches = get_batch_sup(dev_features, hp.eval_batch_size, shuffle=False)
    total_acc = 0.0
    total_loss = 0.0
    num_eval_batches = 0
    for features in range(dev_batches):
        num_eval_batches += 1
        feed_dict = m.create_feed_dict_dev(features, True)
        dev_loss, dev_acc, _gs = sess.run([m.loss, m.acc, m.global_step], feed_dict=feed_dict)

        total_acc += dev_acc
        total_loss += dev_loss
    return total_loss / num_eval_batches, total_acc / num_eval_batches


print("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
rng = random.Random(hp.rand_seed)

print("# Load model")
m = toy_uda(hp)

if not os.path.exists(hp.train_prepro):
    print("# Prepare train/eval data")
    # inputs_a, a_lens, related_labels
    sup_features = process_file_sup(hp.sup_train, hp.vocab, hp.maxlen, rng)
    dev_features = process_file_sup(hp.dev, hp.vocab, hp.maxlen, rng)
    # ori_input, ori_lens, aug_input, aug_lens
    unsup_features = process_file_unsup(hp.unsup_train, hp.vocab, hp.maxlen, hp.maxsentences)

    print("save training data~~~~")
    pickle.dump(sup_features, open(hp.train_prepro, 'wb'))
    pickle.dump(dev_features, open(hp.dev_prepro, 'wb'))
    pickle.dump(unsup_features, open(hp.dev_prepro, 'wb'))

else:
    print("extract training data~~~~")
    sup_features = pickle.load(open(hp.sup_prepro, 'rb'))
    dev_features = pickle.load(open(hp.dev_prepro, 'rb'))
    unsup_features = pickle.load(open(hp.unsup_prepro, 'rb'))

print("# Load model")
m = toy_uda(hp)

print("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.modeldir)
    if ckpt is None:
        print("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.modeldir, "specs"))
    else:
        saver.restore(sess, ckpt)
    m.model = True
    _gs = sess.run(m.global_step)

    tolerant = 0
    for epoch in range(hp.num_epochs):
        total_loss = 0.0
        total_acc = 0.0
        print("<<<<<<<<<<<<<<<< epoch {} >>>>>>>>>>>>>>>>".format(epoch))

        sup_len = len(sup_features)
        sup_batch_num = calc_num_batches(sup_len, hp.sup_batch_size)
        unsup_len = len(unsup_features)
        unsup_batch_num = calc_num_batches(unsup_len, hp.sup_batch_size * hp.unsup_ratio)

        assert unsup_batch_num >= sup_batch_num
        print("sup_batch_num: ", sup_batch_num)
        print("unsup_batch_num: ", unsup_batch_num)

        sup_batches = get_batch_sup(sup_features, hp.sup_batch_size, shuffle=True)
        unsup_batches = get_batch_unsup(unsup_features, hp.sup_batch_size * hp.unsup_ratio, shuffle=True)

        batch_count = 0
        for index in tqdm(range(sup_batch_num)):
            #inputs_a, a_lens, related_labels
            sup_features = sup_batches[index]
            #ori_input, ori_lens, aug_input, aug_lens
            unsup_features = unsup_batches[index]

            batch_count += 1
            #input_sup, sup_len, truth, input_ori, ori_len, input_aug, aug_len, is_training
            feed_dict = m.create_feed_dict(sup_features + unsup_features, True)
            _, _loss, _acc, _gs = sess.run([m.train, m.loss, m.acc, m.global_step], feed_dict=feed_dict)

            total_loss += _loss
            total_acc += _acc

            if batch_count and batch_count % 500 == 0:
                print("batch {:d}: total_loss {:.4f}, acc {:.3f} \n".format(
                    batch_count, _loss, _acc))

        print("\n")
        print("<<<<<<<<<< epoch {} is done >>>>>>>>>>".format(epoch))
        print("# train results")
        train_loss = total_loss / batch_count
        acc = total_acc / batch_count

        print("训练集: total_loss {:.4f}, acc {:.3f} \n".format(train_loss, acc))
        # 验证集
        dev_loss, dev_acc = evaluate(sess, dev_features)
        print("\n")
        print("# evaluation results")
        print("验证集: total_loss {:.4f}, task1_loss {:.4f}, task2_loss {:.4f}, acc {:.3f} \n".format(dev_loss,
                                                                                                   dev_task1_loss,
                                                                                                   dev_task2_loss,
                                                                                                   dev_task2_acc))

        # save model each epoch
        print("#########SAVE MODEL###########")
        model_output = hp.model_path % (epoch, dev_loss, dev_task2_acc)
        ckpt_name = os.path.join(hp.modeldir, model_output)
        saver.save(sess, ckpt_name, global_step=_gs)
        print("training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

print("Done")
