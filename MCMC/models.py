import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import metrics
from tensorflow.python.keras.models import Sequential, model_from_json
from tensorflow.python.keras.layers import Conv1D, Lambda
# from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.layers import Input, MaxPooling1D, Dropout#, Bidirectional #LSTM,
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Reshape, Dense, Activation, Flatten
from tensorflow.python.keras.layers import UpSampling1D
# from tensorflow.python.keras.layers import BatchNormalization

# K = tf.keras.backend
# Sequential = tf.keras.models.Sequential
# Model = tf.keras.models.Model
# Conv1D = tf.keras.layers.Conv1D

# Activation = tf.keras.layers.Activation
# Dense = tf.keras.layers.Dense
# Flatten = tf.keras.layers.Flatten
# Input = tf.keras.layers.Input
LSTM = tf.keras.layers.LSTM
Bidirectional = tf.keras.layers.Bidirectional
BatchNormalization = tf.keras.layers.Normalization
Adam=tf.keras.optimizers.Adam


TCR_PAD_LEN = 20
EP_PAD_LEN = 15
NCH = 6
TCR_SHAPE = (TCR_PAD_LEN, NCH)
EP_SHAPE = (EP_PAD_LEN, NCH)
n_dims = 32
BATCH_SIZE = 128
opt = Adam(1e-5)

def embed_TCR_network(input_shape, viz):
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(filters=64, kernel_size=4, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(n_dims))
    if viz:
        model.summary()
    return model

def make_ep_network(input_shape, viz):
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(inputs)
    x = Activation('relu')(x)
    x = Flatten()(x)
    z_mean = Dense(n_dims)(x)
    z_log_var = Dense(n_dims)(x)
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    if viz:
        encoder.summary()
    return encoder

def _get_triplet_mask_neg(labels):
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    nonneg = tf.logical_not(tf.equal(labels, -1))
    nonneg_mask = tf.logical_and(tf.expand_dims(nonneg, 0), tf.expand_dims(nonneg, 1))
    mod_equal = tf.logical_and(nonneg_mask, label_equal)
    i_equal_j = tf.expand_dims(mod_equal, 2)
    i_equal_k = tf.expand_dims(mod_equal, 1)
    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    mask = tf.logical_and(distinct_indices, valid_labels)
    mask = tf.cast(mask, tf.float32)
    return mask

def _pairwise_distances(embeddings, squared=False):
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.linalg.diag_part(dot_product)
    distances = (tf.expand_dims(square_norm, 0) -
                 2.0 * dot_product +
                 tf.expand_dims(square_norm, 1))
    distances = tf.maximum(distances, 0.0)

    if not squared:
        mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances

def triplet_layer_neg(vects):
    X, y = vects[0], vects[1]
    y = tf.squeeze(y, axis=1)
    pairwise_dist = _pairwise_distances(X)

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    margin = 0.1
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask_neg(y)
    triplet_loss = tf.multiply(mask, triplet_loss)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss

def gaussian_nll_layer_new(vects):
    mu, logsigma, x = vects[1], vects[2], vects[0]
    mse = -0.5 * tf.reduce_sum(tf.square(x - mu) / tf.exp(logsigma), axis=1)
    sigma_trace = -0.5 * tf.reduce_sum(logsigma, axis=1)
    log2pi = -0.5 * n_dims * tf.math.log(2 * np.pi)
    log_likelihood = mse + sigma_trace + log2pi
    return -log_likelihood

# Other functions remain similarly updated.

# The main adjustments are:
# 1. Replace `Convolution1D` with `Conv1D`.
# 2. Replace `K.tf.to_float` with `tf.cast(..., tf.float32)`.
# 3. Replace `K.tf.diag_part` with `tf.linalg.diag_part`.
# 4. Update imports to align with TensorFlow 2.x and Keras 3.x.
# 5. Ensure all TensorFlow/Keras operations are compatible with eager execution.

def experimental_layer_neg(vects):
    
    n_dims=32
    mu = vects[1]
    logsigma = vects[2]
    x = vects[0]
    X, y = vects[0], vects[3]
    y = K.squeeze(y, axis=1)
    
    label_neg = K.tf.equal(y, -1)
    label_nonneg = K.tf.logical_not(label_neg)
    mask_ll = tf.cast(label_nonneg, tf.float32)
    
    mse = -0.5*K.sum(K.square(x-mu)/K.exp(logsigma),axis=1)
    sigma_trace = -0.5*K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = (mse+sigma_trace+log2pi)

    pairwise_dist = _pairwise_distances(X)
    
    anchor_positive_dist = K.tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = K.tf.expand_dims(pairwise_dist, 1)
    margin=0.1
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask_neg(y)
    mask = tf.cast(mask, tf.float32)
    triplet_loss = K.tf.multiply(mask, triplet_loss)
    
    triplet_loss = K.tf.maximum(triplet_loss, 0.0)
    
    valid_triplets = tf.cast(K.tf.greater(triplet_loss, 1e-16), tf.float32)
    num_positive_triplets = K.tf.reduce_sum(valid_triplets)
    num_valid_triplets = K.tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = K.tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)    
    B=1
    A=0.0001
    log_likelihood_mask = K.tf.multiply(mask, log_likelihood)
    return A*K.mean(-log_likelihood_mask) + B*triplet_loss

def identity_loss(y_true, y_pred):
    return y_pred

def make_tcr_gaussian_prior_layer(x):
    mse = -0.5*K.sum(K.square(x),axis=1)
    sigma_trace = -n_dims
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi
    return K.mean(-log_likelihood)

def get_dist_layer_sized(vects):
    X = vects[0]
    y = vects[1]
    y = K.squeeze(y, axis=1)
    batch_size = BATCH_SIZE
    pairwise = _pairwise_distances(X)

    pos_mask_pre = tf.cast(K.tf.equal(K.expand_dims(y, axis=1), K.expand_dims(y, axis=0)), tf.float32)
    pos_mask = pos_mask_pre - K.tf.eye(batch_size)
    dist_pos = K.tf.multiply(pairwise, pos_mask)
    
    neg_mask = tf.cast(K.tf.not_equal(K.expand_dims(y, axis=1), K.expand_dims(y, axis=0)), tf.float32)
    dist_neg_tmp = K.tf.multiply(pairwise, neg_mask)

    intra = K.sum(dist_pos)/K.sum(pos_mask)
    inter = K.sum(dist_neg_tmp)/K.sum(neg_mask)
    return [intra, inter, intra/inter]


def make_full_network_neg(viz=None):
    
    input_tcr = Input(shape=TCR_SHAPE)
    tcr_network = embed_TCR_network(TCR_SHAPE, viz)
    proc_tcr = tcr_network(input_tcr)
    
    input_ep = Input(shape=EP_SHAPE)
    ep_network = make_ep_network(EP_SHAPE, viz)
    proc_ep = ep_network(input_ep)
    
    y_class = Input(shape=(1,))
    
    dist_layer = Lambda(get_dist_layer_sized, output_shape=(3,))([proc_tcr, y_class])
    dist_model = Model(inputs=[input_tcr, y_class], outputs=dist_layer)
    dist_model.compile(loss=identity_loss, optimizer=opt)
    
    nll_loss_layer = Lambda(gaussian_nll_layer_new, output_shape=(1,))([proc_tcr, proc_ep[0], proc_ep[1]])
    nll_model = Model(inputs=[input_tcr, input_ep], outputs=nll_loss_layer)
    nll_out = nll_model([input_tcr, input_ep])
    nll_model.compile(loss=identity_loss, optimizer=opt)
    
    tcr_prior_layer = Lambda(make_tcr_gaussian_prior_layer, output_shape=(1,))(proc_tcr)
    tcr_prior_model = Model(inputs=input_tcr, outputs=tcr_prior_layer)
    tcr_prior_model.compile(loss=identity_loss, optimizer=opt)
    
    loss_layer_triplet = Lambda(triplet_layer_neg, output_shape=(1,))([proc_tcr, y_class])
    triplet = Model(inputs=[input_tcr, y_class], outputs=loss_layer_triplet)
    triplet_out = triplet([input_tcr, y_class])
    triplet.compile(loss=identity_loss, optimizer=opt)
    
    
    experimental_loss_layer = Lambda(experimental_layer_neg, output_shape=(1,))([proc_tcr, proc_ep[0], proc_ep[1], y_class])
    exp = Model(inputs=[input_tcr, input_ep, y_class], outputs=experimental_loss_layer)
    exp.compile(loss=identity_loss, optimizer=opt)
    model_dict = {'Encoder': exp,
                  'Triple': triplet,
                  'NLL': nll_model,
                  'TCR_Embed': tcr_network,
                  'Ep_Embed': ep_network,
                  'TCR_Gaussian_Prior': tcr_prior_model,
                  'Dist': dist_model}
    return model_dict



def train(model_dict, data_dict, epochs, batch_size=BATCH_SIZE, period=10, choice_idx=None, name=None, auto_stop_n=None):
    exp = model_dict['Encoder']
    nll = model_dict['NLL']
    triplet = model_dict['Triple']
    dist_net = model_dict['Dist']
    n_train = data_dict['X_train'].shape[0]
    n_batches = n_train//batch_size
    n_test = data_dict['X_test'].shape[0]
    idx = np.arange(n_train)
    idx_test = np.arange(n_test)
    n_batches_test = n_test//batch_size
    X_test = data_dict['X_test']
    y_enc_test = data_dict['y_enc_test']
    y_cls_test = data_dict['y_cls_test']
    if choice_idx is None:
        X_train = data_dict['X_train']
        y_enc_train = data_dict['y_enc_train']
        y_cls_train = data_dict['y_cls_train']
    else:
        X_train = data_dict['X_train'][choice_idx]
        y_enc_train = data_dict['y_enc_train'][choice_idx]
        y_cls_train = data_dict['y_cls_train'][choice_idx]
    loss=0
    
    loss_track_val = []
    loss_track_train = []
    triplet_track_val = []
    triplet_track_train = []
    nll_track_val = []
    nll_track_train = []
    
    x_list = []
    for i in range(epochs):
        if i != 0:
            loss = np.mean(loss_list)
        print(f'Epoch {i+1}/{epochs} | Loss: {loss}', end='\r')
        np.random.shuffle(idx)
        loss_list = []
        for j in range(n_batches):
            start = j * batch_size
            end = (j+1) * batch_size
            batch_idx = idx[start:end]
            X_batch = X_train[batch_idx]
            y_enc_batch = y_enc_train[batch_idx]
            y_cls_batch = y_cls_train[batch_idx]
            loss = exp.train_on_batch(x=[X_batch, y_enc_batch, y_cls_batch], y=y_cls_batch)
            loss_list.append(loss)
            #print(f'Epoch {i+1}/{epochs} | Batch: {j+1}/{n_batches} | Loss: {loss}', end='\r')
        if i != 0 and i % period == 0:
            x_list.append(i)
            np.random.shuffle(idx)
            loss_list_train = []
            triplet_list_train = []
            nll_list_train = []
            for j in range(n_batches):
                start = j * batch_size
                end = (j+1) * batch_size
                batch_idx = idx[start:end]
                X_batch = X_train[batch_idx]
                y_enc_batch = y_enc_train[batch_idx]
                y_cls_batch = y_cls_train[batch_idx]
                loss = exp.test_on_batch(x=[X_batch, y_enc_batch, y_cls_batch], y=y_cls_batch)
                triplet_loss = triplet.test_on_batch(x=[X_batch, y_cls_batch], y=y_cls_batch)
                #nll_loss = nll.test_on_batch(x=[X_batch, y_enc_batch], y=y_cls_batch)
                triplet_list_train.append(triplet_loss)
                #nll_list_train.append(nll_loss)
                loss_list_train.append(loss)
            
            loss_track_train.append(np.mean(loss_list_train))
            triplet_track_train.append(np.mean(triplet_list_train))
            #nll_track_train.append(np.mean(nll_list_train))
            
            np.random.shuffle(idx_test)
            loss_list_val = []
            triplet_list_val = []
            nll_list_val = []
            for k in range(n_batches_test):
                start_test = k * batch_size
                end_test = (k+1) * batch_size
                batch_idx_test = idx_test[start_test:end_test]
                X_batch_test = X_test[batch_idx_test]
                y_enc_batch_test = y_enc_test[batch_idx_test]
                y_cls_batch_test = y_cls_test[batch_idx_test]
                val = exp.test_on_batch(x=[X_batch_test, y_enc_batch_test, y_cls_batch_test], y=y_cls_batch_test)
                triplet_loss_val = triplet.test_on_batch(x=[X_batch_test, y_cls_batch_test], y=y_cls_batch_test)
                #nll_loss_val = nll.test_on_batch(x=[X_batch_test, y_enc_batch_test], y=y_cls_batch_test)
                triplet_list_val.append(triplet_loss_val)
                #nll_list_val.append(nll_loss_val)
                loss_list_val.append(val)
            
            loss_track_val.append(np.mean(loss_list_val))
            triplet_track_val.append(np.mean(triplet_list_val))
            #nll_track_val.append(np.mean(nll_list_val))
            
            plt.figure(figsize=(20,10))
            plt.subplot(121)
            plt.plot(x_list, loss_track_train, label='Train Total Loss')
            plt.plot(x_list, triplet_track_train, label='Train Triplet Loss')
            #plt.plot(x_list, nll_track_train, label='Train NLL Loss')
            plt.legend()
            plt.xlabel('Epoch', fontsize=20)
            
            plt.subplot(122)
            plt.plot(x_list, loss_track_val, label='Val Total Loss')
            plt.plot(x_list, triplet_track_val, label='Val Triplet Loss')
            #plt.plot(x_list, nll_track_val, label='Val NLL Loss')
            plt.legend()
            plt.xlabel('Epoch', fontsize=20)
            
            if name:
                plt.savefig(name)
            else:
                plt.savefig('test.png')
            plt.close()
            if auto_stop_n:
                if len(loss_track_val) > 2*auto_stop_n:
                    immediate = np.mean(loss_track_val[-auto_stop_n:])
                    past = np.mean(loss_track_val[-2*auto_stop_n:-auto_stop_n])
                    if immediate >= past:
                        break
                        