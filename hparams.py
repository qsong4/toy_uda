import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--maxsentences', default=20000, type=int)

    # train
    ## files
    parser.add_argument('--train', default='./data/snli_train.tsv',
                             help="training data")
    parser.add_argument('--eval', default='./data/snli_test.tsv',
                             help="evaluation data")
    parser.add_argument('--sup_prepro', default='./data/sup_prepro',
                        help="evaluation data")
    parser.add_argument('--unsup_prepro', default='./data/unsup_preprol',
                        help="evaluation data")
    parser.add_argument('--dev_prepro', default='./data/dev_prepro',
                        help="evaluation data")

    parser.add_argument('--model_path', default='FImatchE%02dL%.3fA%.3f')
    parser.add_argument('--modeldir', default='./model')
    parser.add_argument('--vec_path', default='./data/vec/snil_trimmed_vec.npy')

    ## vocabulary
    parser.add_argument('--vocab', default='./data/vocab/snli.vocab',
                        help="vocabulary file path")
    parser.add_argument('--uda_coeff', default=1, type=int, help="uda_coeff")
    parser.add_argument('--rand_seed', default=123, type=int)
    # training scheme
    parser.add_argument('--sup_batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)
    parser.add_argument('--preembedding', default=False, type=bool) #本地测试使用
    parser.add_argument('--uda_softmax_temp', default=1, type=int) #实际训练使用
    parser.add_argument('--unsup_ratio', default=7, type=int)


    #learning rate 0.0003 is too high
    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--num_epochs', default=40, type=int)

    # model
    # This is also the word embedding size , and must can divide by head num.
    parser.add_argument('--d_model', default=300, type=int,
                        help="hidden dimension of interativate")
    parser.add_argument('--d_ff', default=512, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=3, type=int,
                        help="number of extract blocks")
    parser.add_argument('--num_heads', default=6, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen', default=50, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--num_class', default=3, type=int,
                        help="number of class")
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--is_training', default=True, type=bool)

    # test
    parser.add_argument('--test_file', default='./data/snli_test.tsv')
