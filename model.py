import tensorflow as tf

from data_load import load_vocab, loadGloVe
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, ln, positional_encoding_bert
from tensorflow.python.ops import nn_ops


class toy_uda:
    """
    xs: tuple of
        x: int32 tensor. (句子长度，)
        x_seqlens. int32 tensor. (句子)
    """

    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token, self.hp.vocab_size = load_vocab(hp.vocab)
        self.embd = None
        if self.hp.preembedding:
            self.embd = loadGloVe(self.hp.vec_path)
        self.embeddings = get_token_embeddings(self.embd, self.hp.vocab_size, self.hp.d_model, zero_pad=False)
        self.input_sup = tf.placeholder(tf.int32, [None, self.hp.maxlen], name="input_sup")
        self.input_ori = tf.placeholder(tf.int32, [None, self.hp.maxlen], name="input_ori")
        self.input_aug = tf.placeholder(tf.int32, [None, self.hp.maxlen], name="input_aug")
        self.sup_len = tf.placeholder(tf.int32, [None])
        self.ori_len = tf.placeholder(tf.int32, [None])
        self.aug_len = tf.placeholder(tf.int32, [None])
        self.truth = tf.placeholder(tf.int32, [None, self.hp.num_class], name="truth")
        self.is_training = tf.placeholder(tf.bool,shape=None, name="is_training")
        self.model = True

        # self.logits_sup, self.logits_ori, self.logits_aug = self._logits_op()
        self.loss = self._loss_op()
        self.acc = self._acc_op()
        self.global_step = self._globalStep_op()
        self.train = self._training_op()

    def create_feed_dict(self, features, is_training=True):
        input_sup, sup_len, truth, input_ori, ori_len, input_aug, aug_len = features

        feed_dict = {
            self.input_sup: input_sup,
            self.input_ori: input_ori,
            self.input_aug: input_aug,
            self.sup_len: sup_len,
            self.ori_len: ori_len,
            self.aug_len: aug_len,
            self.truth: truth,
            self.is_training: is_training,
        }

        return feed_dict

    def create_feed_dict_dev(self, features, is_training=False):
        input_sup, sup_len, truth = features

        feed_dict = {
            self.input_sup: input_sup,
            self.sup_len: sup_len,
            self.truth: truth,
            self.is_training: is_training,
        }

        return feed_dict

    def kl_for_log_probs(self, log_p, log_q):
        p = tf.exp(log_p)
        neg_ent = tf.reduce_sum(p * log_p, axis=-1)
        neg_cross_ent = tf.reduce_sum(p * log_q, axis=-1)
        kl = neg_ent - neg_cross_ent
        return kl

    def get_tsa_threshold(self, schedule, global_step, num_train_steps, start, end):
        training_progress = tf.to_float(global_step) / tf.to_float(num_train_steps)
        if schedule == "linear_schedule":
            threshold = training_progress
        elif schedule == "exp_schedule":
            scale = 5
            threshold = tf.exp((training_progress - 1) * scale)
            # [exp(-5), exp(0)] = [1e-2, 1]
        elif schedule == "log_schedule":
            scale = 5
            # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
            threshold = 1 - tf.exp((-training_progress) * scale)
        return threshold * (end - start) + start


    def pre_encoder(self, x):
        with tf.variable_scope("pre_encoder", reuse=tf.AUTO_REUSE):
            #x, seqlens, sents1 = xs

            # src_masks
            src_masks = tf.math.equal(x, 0) # (N, T1)

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale

            enc += positional_encoding(enc, self.hp.maxlen)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=self.is_training)

            return enc, src_masks


    def encode(self, encx, src_masks):

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # all_layer = []
            ## Blocks
            for i in range(self.hp.num_blocks_encoder):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    encx = multihead_attention(queries=encx,
                                              keys=encx,
                                              values=encx,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=self.is_training,
                                              causality=False)
                    # feed forward
                    encx = ff(encx, num_units=[self.hp.d_ff, self.hp.d_model])

                    # all_layer.append(encx)

        return encx

    def representation(self):
        pre_sup, sup_mask = self.pre_encoder(self.input_sup)
        input_ori, ori_mask = self.pre_encoder(self.input_ori)
        input_aug, aug_mask = self.pre_encoder(self.input_aug)

        pre_sup = self.encode(pre_sup, sup_mask)
        input_ori = self.encode(input_ori, ori_mask)
        input_aug = self.encode(input_aug, aug_mask)

        return pre_sup, input_ori, input_aug

    def fc(self, inpt, match_dim, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("fc", reuse=reuse):
            w = tf.get_variable("w", [match_dim, self.hp.num_class], dtype=tf.float32)
            b = tf.get_variable("b", [self.hp.num_class], dtype=tf.float32)
            logits = tf.matmul(inpt, w) + b
        # prob = tf.nn.softmax(logits)

        # gold_matrix = tf.one_hot(labels, self.hp.num_class, dtype=tf.float32)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return logits

    def fc_2l(self, inputs, num_units, scope="fc_2l"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Inner layer
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

            # Outer layer
            outputs = tf.layers.dense(outputs, num_units[1])

        return outputs
    # calculate classification accuracy
    def _acc_op(self):
        with tf.name_scope('acc'):
            label_pred = tf.argmax(self.logits_sup, 1, name='label_pred')
            label_true = tf.argmax(self.truth, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    def _logits_op(self):
        # representation
        pre_sup, input_ori, input_aug = self.representation()  # (layers, batchsize, maxlen, d_model)

        #logits = self.fc(agg_res, match_dim=agg_res.shape.as_list()[-1])
        logits_sup = self.fc_2l(pre_sup, num_units=[self.hp.d_ff, self.hp.num_class])
        logits_ori = self.fc_2l(input_ori, num_units=[self.hp.d_ff, self.hp.num_class])
        logits_aug = self.fc_2l(input_aug, num_units=[self.hp.d_ff, self.hp.num_class])
        return logits_sup, logits_ori, logits_aug

    def _log_probs_dev(self):
        sup_probs = tf.nn.log_softmax(self.logits_sup, axis=-1)

        return sup_probs

    def _log_probs(self):
        sup_probs = tf.nn.log_softmax(self.logits_sup, axis=-1)
        ori_probs = tf.nn.log_softmax(self.logits_ori, axis=-1)
        aug_probs = tf.nn.log_softmax(self.logits_aug, axis=-1)

        return sup_probs, ori_probs, aug_probs

    def _loss_op(self):

        if self.model:
            pre_sup, sup_mask = self.pre_encoder(self.input_sup)
            input_ori, ori_mask = self.pre_encoder(self.input_ori)
            input_aug, aug_mask = self.pre_encoder(self.input_aug)

            pre_sup = self.encode(pre_sup, sup_mask)
            input_ori = self.encode(input_ori, ori_mask)
            input_aug = self.encode(input_aug, aug_mask)

            logits_sup = self.fc_2l(pre_sup, num_units=[self.hp.d_ff, self.hp.num_class])
            logits_ori = self.fc_2l(input_ori, num_units=[self.hp.d_ff, self.hp.num_class])
            logits_aug = self.fc_2l(input_aug, num_units=[self.hp.d_ff, self.hp.num_class])

            sup_log_probs, ori_log_probs, aug_log_probs = self._log_probs(logits_sup, logits_ori, logits_aug)
        else:
            pre_sup, sup_mask = self.pre_encoder(self.input_sup)
            pre_sup = self.encode(pre_sup, sup_mask)
            logits_sup = self.fc_2l(pre_sup, num_units=[self.hp.d_ff, self.hp.num_class])

            sup_log_probs = self._log_probs_dev(logits_sup)


        with tf.variable_scope("sup_loss"):

            one_hot_labels = tf.one_hot(self.truth, depth=self.hp.num_class, dtype=tf.float32)
            tgt_label_prob = one_hot_labels

            per_example_loss = -tf.reduce_sum(tgt_label_prob * sup_log_probs, axis=-1)
            loss_mask = tf.ones_like(per_example_loss, dtype=per_example_loss.dtype)
            correct_label_probs = tf.reduce_sum(
                one_hot_labels * tf.exp(sup_log_probs), axis=-1)

            if self.hp.tsa:
                tsa_start = 1. / self.hp.num_class
                tsa_threshold = self.get_tsa_threshold(
                    self.hp.tsa, self.global_step, self.hp.epoch * self.hp.batch_size,
                    tsa_start, end=1)

                larger_than_threshold = tf.greater(
                    correct_label_probs, tsa_threshold)
                loss_mask = loss_mask * (1 - tf.cast(larger_than_threshold, tf.float32))
            else:
                tsa_threshold = 1

            loss_mask = tf.stop_gradient(loss_mask)
            per_example_loss = per_example_loss * loss_mask
            sup_loss = (tf.reduce_sum(per_example_loss) /
                        tf.maximum(tf.reduce_sum(loss_mask), 1))

        unsup_loss_mask = None
        if self.model and self.hp.unsup_ratio > 0:
            with tf.variable_scope("unsup_loss"):

                unsup_loss_mask = 1
                if self.hp.uda_softmax_temp != -1:
                    tgt_ori_log_probs = tf.nn.log_softmax(
                        logits_ori / self.hp.uda_softmax_temp,
                        axis=-1)
                    tgt_ori_log_probs = tf.stop_gradient(tgt_ori_log_probs)
                else:
                    tgt_ori_log_probs = tf.stop_gradient(ori_log_probs)

                if self.hp.uda_confidence_thresh != -1:
                    largest_prob = tf.reduce_max(tf.exp(ori_log_probs), axis=-1)
                    unsup_loss_mask = tf.cast(tf.greater(
                        largest_prob, self.hp.uda_confidence_thresh), tf.float32)
                    unsup_loss_mask = tf.stop_gradient(unsup_loss_mask)

                per_example_kl_loss = self.kl_for_log_probs(
                    tgt_ori_log_probs, aug_log_probs) * unsup_loss_mask
                unsup_loss = tf.reduce_mean(per_example_kl_loss)

        else:
            unsup_loss = 0.

        total_loss = sup_loss

        if self.hp.unsup_ratio > 0 and self.hp.uda_coeff > 0:
            total_loss += self.hp.uda_coeff * unsup_loss

        return total_loss

    def _acc_op(self):
        with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
            label_pred = tf.argmax(self.logits, 1, name='label_pred')
            label_true = tf.argmax(self.truth, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    def _globalStep_op(self):
        global_step = tf.train.get_or_create_global_step()
        return global_step

    def _training_op(self):
        # train scheme
        # global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(self.hp.lr)
        # optimizer = tf.train.AdadeltaOptimizer(lr)

        '''
        if self.hp.lambda_l2>0.0:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            loss = loss + self.hp.lambda_l2 * l2_loss
        '''

        # grads = self.compute_gradients(loss, tvars)
        # grads, _ = tf.clip_by_global_norm(grads, 10.0)
        # train_op = optimizer.minimize(loss, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # train_op = optimizer.minimize(loss)
            train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        return train_op
