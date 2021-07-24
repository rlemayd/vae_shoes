"""
author: Jose M. Saavedra
junio 2021
"""
import tensorflow as tf


def conv3x3(channels, stride = 1, **kwargs):
    return tf.keras.layers.Conv2D(channels, (3,3), 
                                  strides = stride, 
                                  padding = 'same', 
                                  kernel_initializer = 'he_normal', 
                                  **kwargs)
    
##component BathNormalization + RELU
class BNReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BNReLU, self).__init__(**kwargs)
        self.bn = tf.keras.layers.BatchNormalization(name = 'bn')
    
    def call(self, inputs, training = True):
        y = tf.keras.activations.relu(self.bn(inputs, training))
        return y    

#convolutional block
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.conv_1 = tf.keras.layers.Conv2D(channels, (3,3), 
                                  strides = 2, 
                                  padding = 'same', 
                                  kernel_initializer = 'he_normal', 
                                  **kwargs)
        self.conv_2 = tf.keras.layers.Conv2D(channels, (3,3), 
                                  strides = 1, 
                                  padding = 'same', 
                                  kernel_initializer = 'he_normal', 
                                  **kwargs)
        
    def call(self, _input):
        y = self.conv_2(self.conv_1(_input))
        return y

#encoder block
class Encoder(tf.keras.layers.Layer):
    def __init__(self, channels, target_dimension, **kwargs):
        super(Encoder, self).__init__(**kwargs)        
        self.conv1 = ConvBlock(channels[0]) #64
        self.bn_relu_1 = BNReLU()
        self.conv2 = ConvBlock(channels[1]) #32
        self.bn_relu_2 = BNReLU()
        self.conv3 = ConvBlock(channels[2]) #16
        self.bn_relu_3 = BNReLU()
        self.conv4 = ConvBlock(channels[3]) #8 x 8  x 64 
        self.bn_relu_4 = BNReLU()
        self.flatten = tf.keras.layers.Flatten()
        self.dense_mu = tf.keras.layers.Dense(target_dimension)
        self.dense_log_var = tf.keras.layers.Dense(target_dimension)
        
        
    def call(self, inputs, training):
        #input = [128,128,1]
        y = self.bn_relu_1(self.conv1(inputs), training) #64x64
        y = self.bn_relu_2(self.conv2(y), training) #32x32
        y = self.bn_relu_3(self.conv3(y), training) #16x16
        y = self.bn_relu_4(self.conv4(y), training) #8
        y = self.flatten(y)  # 8x8x64 = 4096
        mu = self.dense_mu(y)
        log_var = self.dense_log_var(y)
        x = tf.concat([mu, log_var], axis = 1) #axis
        return x # [mu logvar]

class Decoder(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense_1 = tf.keras.layers.Dense(4096)
        #self.up = tf.keras.layers.UpSampling2D(interpolation = 'bilinear')        
        
        self.conv1 = tf.keras.layers.Conv2DTranspose(channels[0], 3, strides = 2, padding = 'same')
        self.bn_relu_1 = BNReLU()
        self.conv2 = tf.keras.layers.Conv2DTranspose(channels[1], 3, strides = 2, padding = 'same')
        self.bn_relu_2 = BNReLU()
        self.conv3 = tf.keras.layers.Conv2DTranspose(channels[2], 3, strides = 2, padding = 'same')
        self.bn_relu_3 = BNReLU()
        self.conv4 = tf.keras.layers.Conv2DTranspose(channels[3], 3, strides = 2, padding = 'same')
        self.bn_relu_4 = BNReLU()
        self.conv5 = tf.keras.layers.Conv2D(1,  (1,1))
        self.sigmoid = tf.keras.activations.sigmoid
        
                
    def call(self, inputs, training):
        y = self.dense_1(inputs)
        y = tf.reshape(y, (-1, 8,8,64))
        y = self.bn_relu_1(self.conv1(y), training) #16
        y = self.bn_relu_2(self.conv2(y), training) #32
        y = self.bn_relu_3(self.conv3(y), training) #64 
        y = self.bn_relu_4(self.conv4(y), training) #128
        y = self.conv5(y)
        y = self.sigmoid(y)
        return y 
    

class VAE(tf.keras.Model):
    def __init__(self, channels, **kwargs):
        super(VAE,self).__init__(**kwargs)
        self.encoder = Encoder(channels, 128)
        self.decoder = Decoder(tf.reverse(channels, [-1]))
        
    def sampling(self, mu_log_var):
        mu, log_var = tf.split(mu_log_var, 2, axis = 1)
        epsilon = tf.random.normal(tf.shape(mu), mean = 0, stddev = 1)
        return mu + tf.math.exp(log_var / 2) * epsilon
          
    def call(self, _input, training):
        mu_log_var = self.encoder(_input, training)
        z = self.sampling(mu_log_var)        
        x = self.decoder(z, training)        
        x = tf.keras.layers.Flatten()(x)
        out = tf.concat([mu_log_var, x], axis = 1)        
        return out
        
