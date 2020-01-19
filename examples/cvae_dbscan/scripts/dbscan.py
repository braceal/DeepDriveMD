import os
import json
import click
import numpy as np
from glob import glob
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from sklearn.cluster import DBSCAN
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
@click.option('-i', '--sim_path', required=True,
              callback=validate_path,
              help='OpenMM simulation path containing output-cm-*.h5 files')
@click.option('-d', '--cm_path', required=True,
              callback=validate_path,
              help='Preprocessed cvae-input h5 file path')
@click.option('-c', '--cvae_path', required=True,
              callback=validate_path,
              help='CVAE model directory path')
@click.option('-o', '--pdb_out_path', required=True,
              callback=validate_path,
              help='Path to outputted simulation restart points')
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
def main(sim_path, cm_path, cvae_path, pdb_out_path,
         eps_path, eps, min_samples, gpu):

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

        best_eps = eps_record.get(encoder_weight_path)

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

        # If the number of outliers is greater than 150, update eps.
        # Each CVAE model has a different optimal eps.
        if len(outliers) > 150:
            eps = eps + 0.05
        else: 
            eps_record[encoder_weight_path] = eps
            break

    # Save the eps for next round of pipeline
    with open(eps_path, 'w') as file:
        json.dump(eps_record, file)

    # TODO: put in shared folder
    # Get list of current outlier pdb files
    outlier_pdb_fnames = sorted(glob(os.path.join(sim_path, 'outlier-*.pdb')))

    # Remove old pdb outliers that are now inside a cluster
    for pdb_fname in outlier_pdb_fnames:
        # Read atoms and coordinates from PDB
        u = Universe(pdb_fname)
        # Select carbon-alpha atoms
        ca = u.select_atoms('name CA')
        # Compute contact matrix
        cm_matrix = (distances.self_distance_array(ca.positions) < 8.0) * 1.0
        # Use autoecoder to generate embedding of contact matrix
        embedding = encoder.embed(cm_matrix)
        # Cluster embedded contact matrix
        cluster_labels = db.fit_predict(embedding)
        # If PDB is not an outlier, remove it from waiting list
        if cluster_labels[0] != -1:
            os.remove(pdb_fname)

    # Get list of simulation trajectory files (Assume all are equal length)
    traj_fnames = sorted(glob(os.path.join(sim_path, 'output-*.dcd')))

    # Get list of simulation PDB files 
    pdb_fnames = sorted(glob(os.path.join(sim_path, 'input-*.pdb')))

    # Get total number of simulations
    sim_count = len(traj_fnames)

    # Get simulation indices and frame number coresponding to outliers
    outlier_indices = map(lambda outlier : divmod(sim_count, outlier), outliers)

    for sim_id, frame in outlier_indices:
        # TODO: and pipeline-id to pdb_out_path
        pdb_fname = os.path.join(pdb_out_path, f'outlier-{sim_id}-{frame}.pdb')
        mda_traj = mda.Universe(pdb_fnames[sim_id], traj_fnames[sim_id])
        pdb = mda.Writer(pdb_fname)
        pdb.write(mda_traj.atoms)


if __name__ == '__main__':
    main()
