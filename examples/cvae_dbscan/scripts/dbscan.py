import os
import json
import click
import numpy as np
from glob import glob
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from molecules.utils import open_h5
from molecules.ml.unsupervised.cluster import optics_clustering
from molecules.ml.unsupervised import (VAE, EncoderConvolution2D, 
                                       DecoderConvolution2D,
                                       EncoderHyperparams,
                                       DecoderHyperparams)
from deepdrive.utils import get_id
from deepdrive.utils.validators import (validate_path, 
                                        validate_positive,
                                        validate_between_zero_and_one)


def generate_embeddings(encoder_hparams_path, encoder_weight_path, cm_path):
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

    return cm_embeddings

def perform_clustering(eps_path, encoder_weight_path, cm_embeddings, min_samples, eps):
    # TODO: if we decide on OPTICS clustering, then remove all eps code


    # If previous epsilon values have been calculated, load the record from disk.
    # eps_record stores a dictionary from cvae_weight path which uniquely identifies
    # a model, to the epsilon value previously calculated for that model.
    with open(eps_path) as file:
        try:
            eps_record = json.load(file)
        except json.decoder.JSONDecodeError:
            eps_record = {}

    best_eps = eps_record.get(encoder_weight_path)

    # If eps_record contains previous eps value then use it, otherwise use default.
    if best_eps:
        eps = best_eps

    outlier_inds, labels = optics_clustering(cm_embeddings, min_samples)
    eps_record[encoder_weight_path] = eps

    # Save the eps for next round of the pipeline
    with open(eps_path, 'w') as file:
        json.dump(eps_record, file)

    return outlier_inds, labels

def write_rewarded_pdbs(rewarded_inds, sim_path, shared_path):
    # Get list of simulation trajectory files (Assume all are equal length (ns))
    traj_fnames = sorted(glob(os.path.join(sim_path, 'output-*.dcd')))

    # Get list of simulation PDB files
    pdb_fnames = sorted(glob(os.path.join(sim_path, 'input-*.pdb')))

    # Get total number of simulations
    sim_count = len(traj_fnames)

    # Get simulation indices and frame number coresponding to outliers
    reward_locs = list(map(lambda outlier: divmod(outlier, sim_count), rewarded_inds))

    # For documentation on mda.Writer methods see:
    #   https://www.mdanalysis.org/mdanalysis/documentation_pages/coordinates/PDB.html
    #   https://www.mdanalysis.org/mdanalysis/_modules/MDAnalysis/coordinates/PDB.html#PDBWriter._update_frame

    for frame, sim_id in reward_locs:
        pdb_fname = os.path.join(shared_path, f'outlier-{sim_id}-{frame}.pdb')
        u = mda.Universe(pdb_fnames[sim_id], traj_fnames[sim_id])
        with mda.Writer(pdb_fname) as writer:
            # Write a single coordinate set to a PDB file
            writer._update_frame(u)
            writer._write_timestep(u.trajectory[frame])


@click.command()
@click.option('-i', '--sim_path', required=True,
              callback=validate_path,
              help='OpenMM simulation path containing *.dcd and *.pdb files')
@click.option('-s', '--shared_path', required=True,
              callback=validate_path,
              help='Path to folder shared between outlier and MD stages.')
@click.option('-d', '--cm_path', required=True,
              callback=validate_path,
              help='Preprocessed cvae-input h5 file path')
@click.option('-c', '--cvae_path', required=True,
              callback=validate_path,
              help='CVAE model directory path')
@click.option('-e', '--eps_path', required=True,
              callback=validate_path,
              help='Path to eps record for DBSCAN. Empty files are valid.')
@click.option('-E', '--eps', default=0.2, type=float,
              callback=validate_between_zero_and_one,
              help='Value of eps in the DBSCAN algorithm')
@click.option('-m', '--min_samples', default=10, type=int,
              callback=validate_positive,
              help='Value of min_samples in the DBSCAN algorithm')
@click.option('-g', '--gpu', default=0, type=int,
              callback=validate_positive,
              help='GPU id')
def main(sim_path, shared_path, cm_path, cvae_path,
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

    # Define paths to best model and hyperparameters
    encoder_hparams_path = os.path.join(cvae_path, f'encoder-hparams-{best_model_id}.pkl')
    encoder_weight_path = os.path.join(cvae_path, f'encoder-weight-{best_model_id}.h5')

    # Generate embeddings for all contact matrices produced during MD stage
    cm_embeddings = generate_embeddings(encoder_hparams_path, 
                                        encoder_weight_path, cm_path)

    # Performs DBSCAN clustering on embeddings
    outlier_inds, labels = perform_clustering(eps_path, encoder_weight_path,
                                              cm_embeddings, min_samples, eps)

    # Write rewarded PDB files to shared path
    write_rewarded_pdbs(outlier_inds, sim_path, shared_path)

if __name__ == '__main__':
    main()
