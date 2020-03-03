import tensorflow as tf

from model import toy_uda
from tqdm import tqdm
from data_load import process_file_sup, process_file_unsup, get_batch_sup, get_batch_unsup
from utils import save_variable_specs
import os
from hparams import Hparams
import pickle
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def evaluate(sess, eval_init_op, num_eval_batches):

    sess.run(eval_init_op)
    total_steps = 1 * num_eval_batches
    total_acc = 0.0
    total_loss = 0.0
    for i in range(total_steps + 1):
        x, y, x_len, y_len, char_x, char_y, char_x_len, char_y_len, labels = sess.run(data_element)
        feed_dict = m.create_feed_dict(x, y, x_len, y_len, labels, False)
        if hp.char_embedding:
            feed_dict = m.create_char_feed_dict(feed_dict, char_x, char_x_len, char_y, char_y_len)

        #dev_acc, dev_loss = sess.run([dev_accuracy_op, dev_loss_op])
        dev_acc, dev_loss = sess.run([m.acc, m.loss], feed_dict=feed_dict)
        #print("xxx", dev_loss)
        total_acc += dev_acc
        total_loss += dev_loss
    return total_loss/num_eval_batches, total_acc/num_eval_batches

print("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
rng = random.Random(hp.rand_seed)

print("# Prepare train/eval batches")

sup_batches = get_batch_sup(sup_features, hp.sup_batch_size, shuffle=True)
unsup_batches = get_batch_unsup(unsup_features, hp.sup_batch_size * hp.unsup_ratio, shuffle=True)


print("# Load model")
m = toy_uda(hp)

if not os.path.exists(hp.train_prepro):
    # inputs_a, a_lens, related_labels
    sup_features = process_file_sup(hp.sup_train, hp.vocab, hp.maxlen, rng)
    dev_features = process_file_sup(hp.dev, hp.vocab, hp.maxlen, rng)
    # ori_input, ori_lens, aug_input, aug_lens
    unsup_features = process_file_unsup(hp.unsup_train, hp.vocab, hp.maxlen, hp.maxsentences)

    print("save training data~~~~")
    pickle.dump(sup_features,open(hp.train_prepro, 'wb'))
    pickle.dump(dev_features, open(hp.dev_prepro, 'wb'))
    pickle.dump(unsup_features,open(hp.dev_prepro, 'wb'))

else:
    print("extract training data~~~~")
    train_features = pickle.load(open(hp.train_prepro, 'rb'))
    eval_features = pickle.load(open(hp.dev_prepro, 'rb'))

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


    _gs = sess.run(m.global_step)

    tolerant = 0
    for epoch in range(hp.num_epochs):
        loss = 0.0

        acc = 0.0
        print("<<<<<<<<<<<<<<<< epoch {} >>>>>>>>>>>>>>>>".format(epoch))

        sup_batches = get_batch_sup(sup_features, hp.sup_batch_size, shuffle=True)
        unsup_batches = get_batch_unsup(unsup_features, hp.sup_batch_size * hp.unsup_ratio, shuffle=True)

        batch_count = 0
        for features in tqdm(sup_batches):
            batch_count += 1
            feed_dict = m.create_feed_dict(features, True)
            _, _t1loss, _t2loss, _loss, _t2acc, _gs= sess.run([m.train, m.loss_task1, m.loss_task2, m.loss_task_all,
                                                               m.acc, m.global_step], feed_dict=feed_dict)


            # label_pred = tf.argmax(_logit, 1, name='label_pred')
            # print(label_pred)
            # print(_logit)
            # print(related_labels)
            # print(_t2acc)

            task1_loss += _t1loss
            task2_loss += _t2loss
            total_loss += _loss
            task2_acc += _t2acc

            if batch_count and batch_count % 500 == 0:
                print("batch {:d}: task1_loss {:.4f}, task2_loss {:.4f}, total_loss {:.4f}, acc {:.3f} \n".format(
                    batch_count, _t1loss, _t2loss, _loss, _t2acc))


        print("\n")
        print("<<<<<<<<<< epoch {} is done >>>>>>>>>>".format(epoch))
        print("# train results")
        train_loss = total_loss/batch_count
        task2_acc = task2_acc/batch_count
        task1_loss = task1_loss/batch_count
        task2_loss = task2_loss/batch_count

        print("训练集: task1_loss {:.4f}, task2_loss {:.4f}, total_loss {:.4f}, acc {:.3f} \n".format(
            task1_loss, task2_loss, train_loss, task2_acc))
        #验证集
        dev_loss, dev_task1_loss, dev_task2_loss, dev_task2_acc = evaluate(sess, eval_features)
        print("\n")
        print("# evaluation results")
        print("验证集: total_loss {:.4f}, task1_loss {:.4f}, task2_loss {:.4f}, acc {:.3f} \n".format(dev_loss, dev_task1_loss, dev_task2_loss, dev_task2_acc))


        # save model each epoch
        print("#########SAVE MODEL###########")
        model_output = hp.model_path % (epoch, dev_loss, dev_task2_acc)
        ckpt_name = os.path.join(hp.modeldir, model_output)
        saver.save(sess, ckpt_name, global_step=_gs)
        print("training of {} epochs, {} has been saved.".format(epoch, ckpt_name))


print("Done")
