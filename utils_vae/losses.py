import tensorflow as tf

def vae_reconstruction_loss(x_true, x_pred):
    #r_loss = tf.reduce_mean(tf.square(x_true-x_pred), axis = [1,2])
    r_loss= tf.reduce_sum(tf.keras.losses.binary_crossentropy(x_true, x_pred), axis = [1,2])
    return r_loss

def vae_kl_loss(mu, log_var):
    kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis = 1)
    return kl_loss

def vae_loss(y_true, y_pred):
    mu_log_var = tf.slice(y_pred, [0,0],[-1,256])
    x = tf.slice(y_pred, [0,256],[-1,-1])        
    # x_pred = tf.reshape(x, (-1, 128,128,1))
    x_pred = tf.reshape(x, (-1, 64, 64, 1))
    mu, log_var = tf.split(mu_log_var, 2, axis = 1)
    r_loss = tf.reduce_mean(vae_reconstruction_loss(y_true, x_pred))
    kl_loss = tf.reduce_mean(vae_kl_loss(mu, log_var))    
    return r_loss + kl_loss

