from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import AveragePooling2D, MaxPool2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model, convert_dense_weights_data_format
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import decode_predictions
import keras.backend as K

DENSENET_121_WEIGHTS_PATH = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-121-32.h5'
DENSENET_161_WEIGHTS_PATH = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-161-48.h5'
DENSENET_169_WEIGHTS_PATH = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-169-32.h5'
DENSENET_121_WEIGHTS_PATH_NO_TOP = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-121-32-no-top.h5'
DENSENET_161_WEIGHTS_PATH_NO_TOP = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-161-48-no-top.h5'
DENSENET_169_WEIGHTS_PATH_NO_TOP = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-169-32-no-top.h5'

def __conv_block(ip, nb_filter, bottlenect=False, dropout_rate=None, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    
    if bottlenect:
        inter_channel = nb_filter * 4
        x = Conv2D(inter_channel, (1,1), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        
    x = Conv2D(nb_filter, (3,3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
        
    return x

def __dense_block(x, nb_layers, nb_filter, growth_rate, bottlenect=False, dropout_rate=None, weight_decay=1e-4,
                  grow_nb_filters=True, return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: keras tensor with nb_layers of conv_block appended
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x_list = [x]
    
    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottlenect, dropout_rate, weight_decay)
        x_list.append(cb)
        
        x = concatenate([x, cb], axis=concat_axis)
        
        if grow_nb_filters:
            nb_filter += growth_rate
            
    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter
    
def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4):
    
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_filter*compression), (1,1), kernel_initializer='he_normal', 
               padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)
    return x

def __create_dense_net(nb_classes, img_input, include_top, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                       nb_layers_per_block=4, bottlenect=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                       subsample_initial_block=False, activation='softmax'):
    ''' Build the DenseNet model
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the depth of the network.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay rate
        subsample_initial_block: Set to True to subsample the initial convolution and
                add a MaxPool2D before the dense blocks are added.
        subsample_initial:
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
    Returns: keras tensor with nb_layers of conv_block appended
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    if reduction != 0.0:
        assert reduction < 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'
    
    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
        
        assert len(nb_layers) == nb_dense_block, 'If list, nb_layer is used as provided. ' \
                                                   'Note that list size must be (nb_dense_block)'
        
        final_nb_layer = nb_layers[-1]
        nb_layers = nb_layers[:-1]
        
    else:
        if nb_layers_per_block == -1:
            assert (depth-4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
            
            count = int((depth-4) / 3)
            
            if bottlenect:
                count = count // 2
            
            nb_layers = [count for _ in range(nb_dense_block)]
            final_nb_layer = count
        else:
            final_nb_layer = nb_layers_per_block
            nb_layers = [nb_layers_per_block for _ in range(nb_dense_block)]
            
    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate
        
    # compute compression factor
    compression = 1.0 - reduction
    
    # Initial convolution
    if subsample_initial_block:
        initial_kernel = (7,7)
        initial_strides= (2,2)
    else:
        initial_kernel = (3,3)
        initial_strides= (1,1)
        
    x = Conv2D(nb_filter, initial_kernel, padding='same', strides=initial_strides, 
               use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)
    
    if subsample_initial_block:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        
    # add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, 
                                     bottlenect=bottlenect, dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition_block
        x = __transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)
        
    # The last dense_block does not have a transition_block
    x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate,
                                 bottlenect=bottlenect, dropout_rate=dropout_rate, weight_decay=weight_decay)
    
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    
    if include_top:
        x = Dense(nb_classes, activation=activation)(x)
        
    return x

def DenseNet_m(input_shape=None, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1, nb_layers_per_block=-1,
             bottlenect=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, subsample_initial_block=False,
             include_top=True, weights=None, input_tensor=None,
             classes=10, activation='softmax'):
    
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `cifar10` '
                         '(pre-training on CIFAR-10).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top`'
                         ' as true, `classes` should be 1000')

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')
    
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape, default_size=32, min_size=8, 
                                      data_format=K.image_data_format(), require_flatten=include_top)
    
    if input_tensor == None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
            
    x = __create_dense_net(classes, img_input, include_top, depth, nb_dense_block, growth_rate, nb_filter,
                       nb_layers_per_block, bottlenect, reduction, dropout_rate, weight_decay,
                       subsample_initial_block, activation)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    # Create model
    model = Model(inputs, x, name='densenet')
    
    # load weights
    if weights == 'imagenet':
        weights_loaded = False

        if (depth == 121) and (nb_dense_block == 4) and (growth_rate == 32) and (nb_filter == 64) and \
                (bottleneck is True) and (reduction == 0.5) and (dropout_rate == 0.0) and (subsample_initial_block):
            if include_top:
                weights_path = get_file('DenseNet-BC-121-32.h5',
                                        DENSENET_121_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='a439dd41aa672aef6daba4ee1fd54abd')
            else:
                weights_path = get_file('DenseNet-BC-121-32-no-top.h5',
                                        DENSENET_121_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='55e62a6358af8a0af0eedf399b5aea99')
            model.load_weights(weights_path)
            weights_loaded = True

        if (depth == 161) and (nb_dense_block == 4) and (growth_rate == 48) and (nb_filter == 96) and \
                (bottleneck is True) and (reduction == 0.5) and (dropout_rate == 0.0) and (subsample_initial_block):
            if include_top:
                weights_path = get_file('DenseNet-BC-161-48.h5',
                                        DENSENET_161_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='6c326cf4fbdb57d31eff04333a23fcca')
            else:
                weights_path = get_file('DenseNet-BC-161-48-no-top.h5',
                                        DENSENET_161_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='1a9476b79f6b7673acaa2769e6427b92')
            model.load_weights(weights_path)
            weights_loaded = True

        if (depth == 169) and (nb_dense_block == 4) and (growth_rate == 32) and (nb_filter == 64) and \
                (bottleneck is True) and (reduction == 0.5) and (dropout_rate == 0.0) and (subsample_initial_block):
            if include_top:
                weights_path = get_file('DenseNet-BC-169-32.h5',
                                        DENSENET_169_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='914869c361303d2e39dec640b4e606a6')
            else:
                weights_path = get_file('DenseNet-BC-169-32-no-top.h5',
                                        DENSENET_169_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='89c19e8276cfd10585d5fadc1df6859e')
            model.load_weights(weights_path)
            weights_loaded = True

        if weights_loaded:
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)

            if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')

            print("Weights for the model were loaded successfully")

    return model

class DenseNet():
    @staticmethod
    def build(width, height, depth, classes):
        # create img_input
        img_dim = (width, height, depth)
        depth = 40
        nb_dense_block = 3
        growth_rate = 12
        nb_filter = -1
        dropout_rate = 0.0 # 0.0 for data augmentation
        model = DenseNet_m(img_dim, classes=classes, depth=depth, nb_dense_block=nb_dense_block,
                          growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None)
        return model
