import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name",
                        default="bert-large-cased-whole-word-masking", type=str)
    parser.add_argument("--output_dir", default="logs", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")

    # Other parameters
    parser.add_argument("--num_workers", default=30, type=int)
    parser.add_argument("--train_file", type=str,
                        default="")
    parser.add_argument("--predict_file", type=str,
                        default="")
    parser.add_argument("--init_checkpoint", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).",
                        default="")
    parser.add_argument("--max_q_length", default=60, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_ans_length", default=20, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", default=False,
                        action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False,
                        action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=8,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=100,
                        type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=1e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=500, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--iterations_per_loop", default=1000, type=int,
                        help="How many steps to make in each estimator call.")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument('--seed', type=int, default=3,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--eval_period', type=int, default=2500)
    parser.add_argument('--verbose', action="store_true", default=False)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_beams", default=1, type=int)

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--use-adam", action="store_true")
    parser.add_argument("--use-adafactor", action="store_true")
    parser.add_argument("--drop", type=float, default=0.1)
    parser.add_argument("--decode-bridge", action="store_true")

    parser.add_argument("--complex", action="store_true", help="useful when evaluating for complexwebquestions")
    
    args = parser.parse_args()

    return args