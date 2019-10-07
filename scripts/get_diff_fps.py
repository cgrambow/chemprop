"""Script to calculate difference fingerprints"""

from argparse import ArgumentParser, Namespace
import pickle
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from chemprop.data import ReactionDataset
from chemprop.data.utils import get_data_from_smiles, get_smiles
from chemprop.models import ReactionModel
from chemprop.utils import load_args, load_checkpoint, makedirs


class Identity(nn.Module):
    """Identity PyTorch module."""
    def forward(self, x, *args, **kwargs):
        return x


def compute_fingerprints(model: nn.Module,
                         data: ReactionDataset,
                         batch_size: int) -> List[np.ndarray]:
    """
    Computes the difference atom fingerprints from a ReactionModel.

    :param model: A ReactionModel.
    :param data: A ReactionDataset.
    :param batch_size: Batch size.
    :return: A list of 2D numpy arrays of size natomsxhidden_size containing
    the difference fingerprints generated by the model for each molecule provided.
    """
    model.eval()
    model.diff_encoder = Identity()
    model.ffn = Identity()

    fps = []
    num_iters, iter_step = len(data), batch_size

    for i in trange(0, num_iters, iter_step):
        # Prepare batch
        data_batch = data[i:i + batch_size]
        mol_batch = ReactionDataset(data_batch)
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch

        with torch.no_grad():
            rbatch, pbatch = list(zip(*batch))
            batch_fps = model(rbatch, pbatch, features_batch)

        # Collect vectors
        batch_fps = batch_fps.data.cpu().numpy()
        batch_fps = batch_fps[1:]  # Remove zero-padding
        fps.append(batch_fps)

    return fps


def get_fingerprints(args: Namespace,
                     smiles: Union[List[str], List[Tuple[str, str]]],
                     model: ReactionModel) -> List:
    """
    For each reaction, compute its difference fingerprints.

    :param args: Arguments.
    :param smiles: A list of tuples of SMILES strings.
    :param model: A trained ReactionModel.
    :return: A list of difference fingerprints.
    """
    data = get_data_from_smiles(smiles, args=args)

    print('Computing fingerprints')
    # Use batch size of 1 so that we get list of fingerprints where each item in list is a molecule
    return compute_fingerprints(model, data, batch_size=1)


def get_fingerprints_from_file(args: Namespace) -> List:
    """
    For each reaction, compute its difference fingerprints.
    Loads reactions and model from file.

    :param args: Arguments.
    :return: A list of difference fingerprints.
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

    return get_fingerprints(args, smiles, model)


def save_fingerprints(args: Namespace):
    """
    For each reaction, compute its difference fingerprints.
    Loads reactions and model from file and saves
    all fingerprints to a file.

    :param args: Arguments.
    """
    fps = get_fingerprints_from_file(args)

    makedirs(args.save_path, isfile=True)
    with open(args.save_path, 'wb') as f:
        pickle.dump(fps, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV file with dataset')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to .pt file containing a model checkpoint')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to .pickle file where fingerprints will be saved')

    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    args.features_generator = None
    args.use_input_features = False

    save_fingerprints(args)
