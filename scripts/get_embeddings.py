"""Script to calculate molecule or reaction embeddings (prior to FFN)"""

from argparse import ArgumentParser, Namespace
from typing import List, Tuple, Union

import numpy as np
import torch

from chemprop.data.utils import get_data_from_smiles, get_smiles
from chemprop.models import MoleculeModel, ReactionModel
from chemprop.nn_utils import compute_embeddings
from chemprop.utils import load_args, load_scalers, load_checkpoint, makedirs


def get_embeddings(args: Namespace,
                   smiles: Union[List[str], List[Tuple[str, str]]],
                   model: Union[MoleculeModel, ReactionModel],
                   batch_size: int = 50) -> np.ndarray:
    """
    For each molecule or reaction, compute its embedding.

    :param args: Arguments.
    :param smiles: A list of (tuples of) SMILES strings.
    :param model: A trained MoleculeModel or ReactionModel..
    :param batch_size: Batch size.
    :return: An array of embeddings.
    """
    data = get_data_from_smiles(smiles, args=args)

    print('Computing embeddings')
    return np.array(compute_embeddings(model, data, batch_size))


def get_embeddings_from_file(args: Namespace) -> np.ndarray:
    """
    For each molecule or reaction, compute its embedding.
    Loads molecules/reactions and model from file.

    :param args: Arguments.
    :return: An array of embeddings.
    """
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    print('Loading training args')
    train_args = load_args(args.checkpoint_path)

    # Update args with training arguments
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print('Loading data')
    smiles = get_smiles(args.data_path, reaction=args.reaction)

    print('Loading model')
    model = load_checkpoint(args.checkpoint_path, current_args=args, cuda=args.cuda)

    return get_embeddings(args, smiles, model, batch_size=args.batch_size)


def save_embeddings(args: Namespace):
    """
    For each molecule or reaction, compute its embedding.
    Loads molecules/reactions and model from file and saves
    all embeddings to a file.

    :param args: Arguments.
    """
    embeddings = get_embeddings_from_file(args)

    makedirs(args.save_path, isfile=True)
    np.save(args.save_path, embeddings)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV file with dataset')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to .pt file containing a model checkpoint')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to .npy file where embeddings will be saved')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size when making predictions')

    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    # For now this might crash when using features and having more than one FFN layer
    args.features_generator = None
    args.use_input_features = False

    save_embeddings(args)
