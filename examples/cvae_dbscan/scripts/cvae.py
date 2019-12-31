import os
import click
from keras.optimizers import RMSprop
from molecules.utils import open_h5
from molecules.ml.unsupervised import (VAE, EncoderConvolution2D, 
                                       DecoderConvolution2D,
                                       HyperparamsEncoder,
                                       HyperparamsDecoder)
from molecules.ml.unsupervised.callbacks import (EmbeddingCallback,
                                                LossHistory)
from deepdrive.utils.validators import validate_path, validate_positive


@click.command()
@click.option('-i', '--input', 'input_path', required=True,
              callback=validate_path,
              help='Path to file containing preprocessed contact matrix data')
@click.option('-o', '--out', 'out_path', required=True,
              callback=validate_path,
              help='Output directory for model data')
@click.option('-m', '--model_id', required=True, type=int,
              callback=validate_positive,
              help='Model ID in pipeline [0...N]')
@click.option('-g', '--gpu', default=0, type=int,
              callback=validate_positive,
              help='GPU id')
@click.option('-e', '--epochs', default=100, type=int,
              callback=validate_positive,
              help='Number of epochs to train for')
@click.option('-b', '--batch_size', default=512, type=int,
              callback=validate_positive,
              help='Batch size for training')
@click.option('-d', '--latent_dim', default=3, type=int,
              callback=validate_positive,
              help='Number of dimensions in latent space')
def main(input_path, out_path, model_id, gpu, epochs, batch_size, latent_dim):

    # Set CUDA environment variables
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    with open_h5(input_path) as input_file:

        # Access contact matrix data from h5 file
        data = input_file['contact_maps']

        # 80-20 train validation split index
        split = int(0.8 * len(data))

        # Partition input data into 80-20 train valid split
        train, valid = data[:split], data[split:]

        # Get shape of an individual contact matrix 
        # (ignore total number of matrices)
        input_shape = train.shape[1:]

        # Set model hyperparameters for encoder and decoder
        shared_hparams = {'num_conv_layers': 4,
                          'filters': [64, 64, 64, 64],
                          'kernels': [3, 3, 3, 3],
                          'strides': [1, 2, 1, 1],
                          'num_affine_layers': 1,
                          'affine_widths': [128],
                          'latent_dim': latent_dim
                         }

        affine_dropouts = [0]

        encoder_hparams = HyperparamsEncoder(affine_dropouts=affine_dropouts,
                                             **shared_hparams)
        decoder_hparams = HyperparamsDecoder(**shared_hparams)

        encoder = EncoderConvolution2D(input_shape=input_shape,
                                       hyperparameters=encoder_hparams)

        # Get shape attributes of the last encoder layer to define the decoder
        encode_conv_shape, num_conv_params = encoder.get_final_conv_params()

        decoder = DecoderConvolution2D(output_shape=input_shape,
                                       enc_conv_params=num_conv_params,
                                       enc_conv_shape=encode_conv_shape,
                                       hyperparameters=decoder_hparams)

        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

        cvae = VAE(input_shape=input_shape,
                   encoder=encoder,
                   decoder=decoder,
                   optimizer=optimizer)

        # Define callbacks to report model performance for analysis
        embed_callback = EmbeddingCallback(train, cvae)
        loss_callback = LossHistory()

        cvae.train(data=train, validation_data=valid, 
                   batch_size=batch_size, epochs=epochs, 
                   callbacks=[embed_callback, loss_callback])

        # Define file paths to store model performance and weights 
        weight_path = os.path.join(out_path, f'weight-{model_id}.h5')
        embed_path = os.path.join(out_path, f'embed-{model_id}.npy')
        idx_path = os.path.join(out_path, f'embed-idx-{model_id}.npy')
        loss_path = os.path.join(out_path, f'loss-{model_id}.npy')
        val_loss_path = os.path.join(out_path, f'val-loss-{model_id}.npy')

        # Save model performance and weights
        cvae.save_weights(weight_path)
        embed_callback.save(embed_path=embed_path, idx_path=idx_path)
        loss_callback.save(loss_path=loss_path, val_loss_path=val_loss_path)


if __name__ == '__main__':
    main()
