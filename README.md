# Variational Autoencoder (VAE)
This is an implementation of a variational autoencoder using convolutional layers. I will show an example of using it with a sketch dataset.

# Dependencies
Project [convnet2](https://github.com/jmsaavedrar/convnet2)

# Training

python train.py -mode train -config configs/sbir_vae.config -name SBIR

# Example of reconstruction

python train.py -mode predict -config configs/sbir_vae.config -name SBIR

