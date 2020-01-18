import os
import json
import click
import numpy as np
from glob import glob
from sklearn.cluster import DBSCAN
from keras.optimizers import RMSprop
from molecules.utils import open_h5
from molecules.ml.unsupervised import (VAE, EncoderConvolution2D, 
                                       DecoderConvolution2D,
                                       EncoderHyperparams,
                                       DecoderHyperparams)
from deepdrive.utils import get_id
from deepdrive.utils.validators import (validate_path, 
                                        validate_positive,
                                        validate_between_zero_and_one)


@click.command()
@click.option('-d', '--cm_path', required=True,
              callback=validate_path,
              help='Preprocessed cvae-input h5 file path')
@click.option('-c', '--cvae_path', required=True,
              callback=validate_path,
              help='CVAE model directory path')
@click.option('-p', '--pdb_path', required=True,
              callback=validate_path, help='Path to PDB file')
@click.option('-o', '--out_path', required=True,
              callback=validate_path,
              help='Path to outputted simulation restart points')
@click.option('-r', '--ref_path', default=None,
              callback=validate_path,
              help='Path to reference pdb for RMSD')
@click.option('-e', '--eps_path', default=None,
              callback=validate_path,
              help='Path to eps record for DBSCAN')
@click.option('-E', '--eps', default=0.2, type=float,
              callback=validate_between_zero_and_one,
              help='Value of eps in the DBSCAN algorithm')
@click.option('-m', '--min_samples', default=10, type=int,
              callback=validate_positive,
              help='Value of min_samples in the DBSCAN algorithm')
@click.option('-g', '--gpu', default=0, type=int,
              callback=validate_positive,
              help='GPU id')
def main(cm_path, cvae_path, pdb_path, out_path,
         ref_path, eps_path, eps, min_samples, gpu):

    # Set CUDA environment variables
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    
    # Identify the latest models with lowest validation loss
    # Gather validation loss reports from each model in the current pipeline round
    # Find the minimum validation loss by taking the model_id associated with
    # the smallest validation loss during the last epoch. 
    best_model_id = get_id(min(glob(os.path.join(cvae_path, 'val-loss-*.npy')),
                               key=lambda loss_path: np.load(loss_path)[-1]), 'npy')


    # Define paths to best model hyperparameters
    encoder_hparams_path = os.path.join(cvae_path, f'encoder-hparams-{best_model_id}.pkl')

    encoder_weight_path = os.path.join(cvae_path, f'encoder-weight-{best_model_id}.h5')

    encoder_hparams = EncoderHyperparams.load(encoder_hparams_path)

    with open_h5(cm_path) as file:

        # Access contact matrix data from h5 file
        data = file['contact_maps']

        # Get shape of an individual contact matrix 
        # (ignore total number of matrices)
        input_shape = data.shape[1:]

        encoder = EncoderConvolution2D(input_shape=input_shape,
                                       hyperparameters=encoder_hparams)

        # Load best model weights
        encoder.load_weights(encoder_weight_path)

        # Create contact matrix embeddings
        cm_embeddings, *_ = encoder.embed(data)

    # If previous epsilon values have been calculated, load the record from disk.
    # eps_record stores a dictionary from cvae_weight path which uniquely identifies
    # a model, to the epsilon value previously calculated for that model.
    if eps_path:
        with open(eps_path) as file:
            eps_record = json.load(file)

        best_eps = eps_record.get(cvae_weight_path)

        if best_eps:
            eps = best_eps

    else:
        eps_record = {}

    # Search for right eps for DBSCAN 
    while True:
        # Run DBSCAN clustering on contact matrix embeddings
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(cm_embeddings)
        # Array of outlier indices in latent space
        outliers = np.flatnonzero(db.labels_ == -1)

        # If the number of outliers is greater than 150, update eps
        if len(outliers) > 150:
            eps = eps + 0.05
        else: 
            eps_record[cvae_weight_path] = eps
            break

    # Save the eps for next round of pipeline
    with open(eps_path, 'w') as file:
        json.dump(eps_record, file)




if __name__ == '__main__':
    main()