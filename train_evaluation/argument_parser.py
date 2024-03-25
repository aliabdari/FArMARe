import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description='Training specs')
    parser.add_argument("--output_feature_size", type=int, default=256, required=False,
                        help='The size of the output feature')
    parser.add_argument("--is_bidirectional", type=bool, default=True, required=False,
                        help='Use the Bidirectional GRU or not')
    parser.add_argument("--is_token_level", type=bool, default=False, required=False,
                        help='using the token level features or not')
    parser.add_argument("--num_epochs", type=int, default=50, required=False, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=64, required=False, help='batch size')
    parser.add_argument("--lr", type=float, default=.008, required=False, help='learning rate')
    parser.add_argument("--step_size", type=int, default=27, required=False,
                        help='Step size for the decay of the learning rate')
    parser.add_argument("--gamma", type=float, default=0.75, required=False,
                        help='learning rate decay factor with which the learning rate will be reduced')
    return parser
