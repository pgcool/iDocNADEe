import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


def vectors(model, data, session):
    '''
    Function for getting document hidden representation for DocNADE

    model:   DocNADE model instance
    data:    data object instance
    session: tensorflow session object
    '''
    vecs = []
    for _, x, seq_lengths in data:
        vecs.extend(
            session.run([model.h], feed_dict={
                model.x: x,
                model.seq_lengths: seq_lengths
            })[0]
        )
    return np.array(vecs)

def vectors_bidirectional(model, data, session, combination_type):
    '''
    Function for getting document hidden representation for iDocNADE

    model:            DocNADE model instance
    data:             data object instance
    session:          tensorflow session object
    combination_type: parameter for combining forward and backward hidden representation for a document
    '''
    vecs = []
    if combination_type == "concat":
        for _, x, x_bw, seq_lengths in data:
            vecs.extend(
                session.run([model.h_comb_concat], feed_dict={
                    model.x: x,
                    model.x_bw: x_bw,
                    model.seq_lengths: seq_lengths
                })[0]
            )
    elif combination_type == "sum":
        for _, x, x_bw, seq_lengths in data:
            vecs.extend(
                session.run([model.h_comb_sum], feed_dict={
                    model.x: x,
                    model.x_bw: x_bw,
                    model.seq_lengths: seq_lengths
                })[0]
            )
    else:
        print("vectors function: Invalid value for combination_type.")
    return np.array(vecs)


def loss(model, data, session):
    loss = []
    for _, x, seq_lengths in data:
        loss.append(
            session.run([model.loss], feed_dict={
                model.x: x,
                model.seq_lengths: seq_lengths
            })[0]
        )
    return sum(loss) / len(loss)


def gradients(opt, loss, vars, step, max_gradient_norm=None, dont_clip=[]):
    '''
    Function for calculating and applying gradients on all trainable parameters
    '''
    gradients = opt.compute_gradients(loss, vars)
    if max_gradient_norm is not None:
        to_clip = [(g, v) for g, v in gradients if v.name not in dont_clip]
        not_clipped = [(g, v) for g, v in gradients if v.name in dont_clip]
        gradients, variables = zip(*to_clip)
        clipped_gradients, _ = clip_ops.clip_by_global_norm(
            gradients,
            max_gradient_norm
        )
        gradients = list(zip(clipped_gradients, variables)) + not_clipped

    # Add histograms for variables, gradients and gradient norms
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is None:
            print('warning: missing gradient: {}'.format(variable.name))
        if grad_values is not None:
            tf.summary.histogram(variable.name, variable)
            tf.summary.histogram(variable.name + '/gradients', grad_values)
            tf.summary.histogram(
                variable.name + '/gradient_norm',
                clip_ops.global_norm([grad_values])
            )

    return opt.apply_gradients(gradients, global_step=step)


def linear(input, output_dim, input_bw=None, scope=None, stddev=None, W_initializer=None):
    '''
    Function to generate softmax distribution over vocabulary for a document during training.

    input:         input batch of documents
    output_dim:    vocabulary size
    input_bw:      input batch of documents for backward direction
    W_initializer: initializer for softmax projection matrix V
    '''
    const = tf.constant_initializer(0.0)

    if W_initializer is None:
        if stddev:
            norm = tf.random_normal_initializer(stddev=stddev)
        else:
            norm = tf.random_normal_initializer(
                stddev=np.sqrt(2.0 / input.get_shape()[1].value)
            )

        with tf.variable_scope(scope or 'linear'):
            w = tf.get_variable(
                'w',
                [input.get_shape()[1], output_dim],
                initializer=norm
            )
    else:
        w = tf.get_variable(
            'w',
            initializer=tf.transpose(W_initializer)
        )

    b = tf.get_variable('b', [output_dim], initializer=const)
    b_bw = tf.get_variable('b_bw', [output_dim], initializer=const)

    input_bw_logits = None
    if input_bw is None:
        input_logits = tf.nn.xw_plus_b(input, w, b)
    else:
        input_logits = tf.nn.xw_plus_b(input, w, b)
        input_bw_logits = tf.nn.xw_plus_b(input_bw, w, b_bw)
    
    return input_logits, input_bw_logits


def linear_reload(input, output_dim, input_bw=None, scope=None, stddev=None, W_initializer=None, 
                    V_reload=None, b_reload=None, b_bw_reload=None):
    '''
    Function to generate softmax distribution over vocabulary for a document during model reload.

    input:         input batch of documents
    output_dim:    vocabulary size
    input_bw:      input batch of documents for backward direction
    V_reload:      parameter to initialize the V matrix (softmax logits)
    b_reload:      parameter to initialize the b bias (softmax logits)
    '''
    w = tf.Variable(
        initial_value=V_reload,
        trainable=False
    )
    b = tf.Variable(
        initial_value=b_reload,
        trainable=False
    )

    if input_bw is None:
        b_bw = None
    else:
        b_bw = tf.Variable(
            initial_value=b_bw_reload,
            trainable=False
        )

    input_bw_logits = None
    if input_bw is None:
        input_logits = tf.nn.xw_plus_b(input, w, b)
    else:
        input_logits = tf.nn.xw_plus_b(input, w, b)
        input_bw_logits = tf.nn.xw_plus_b(input_bw, w, b_bw)
    
    return input_logits, input_bw_logits


def masked_sequence_cross_entropy_loss(
    x,
    seq_lengths,
    logits,
    loss_function=None,
    norm_by_seq_lengths=True
):
    '''
    Compute the cross-entropy loss between all elements in x and logits.
    Masks out the loss for all positions greater than the sequence
    length (as we expect that sequences may be padded).

    Optionally, also either use a different loss function (eg: sampled
    softmax), and/or normalise the loss for each sequence by the
    sequence length.
    '''
    batch_size = tf.shape(x)[0]
    labels = tf.reshape(x, [-1])

    max_doc_length = tf.reduce_max(seq_lengths)
    mask = tf.less(
        tf.range(0, max_doc_length, 1),
        tf.reshape(seq_lengths, [batch_size, 1])
    )
    mask = tf.reshape(mask, [-1])
    mask = tf.to_float(tf.where(
        mask,
        tf.ones_like(labels, dtype=tf.float32),
        tf.zeros_like(labels, dtype=tf.float32)
    ))

    if loss_function is None:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels
        )
    else:
        loss = loss_function(logits, labels)
    loss *= mask
    loss = tf.reshape(loss, [batch_size, -1])
    loss = tf.reduce_sum(loss, axis=1)
    loss_unnormed = loss
    if norm_by_seq_lengths:
        loss = loss / tf.to_float(seq_lengths)
    return tf.reduce_mean(loss), labels, mask, tf.reduce_mean(loss_unnormed)


class DocNADE(object):
    '''
    Python class for creating DocNADE object.
    '''
    def __init__(self, x, y, seq_lengths, params,
                 W_initializer=None, lambda_embeddings=-1.0,
                 W_pretrained=None):
        '''
        x:           input of shape [batch_size X max document length] contains word indices
        y:           document labels
        seq_lengths: actual sequence lengths of documents in the padded matrix input x
        params:      training comfiguration parameters
        '''
        self.x = x
        self.y = y
        self.seq_lengths = seq_lengths

        batch_size = tf.shape(x)[0]
        self.b_s = tf.shape(x)
        self.initializer_embeddings = W_initializer
        self.lambda_embeddings = lambda_embeddings

        # Create W (embedding matrix) of DocNADE
        with tf.device('/cpu:0'):
            if W_pretrained is None:
                max_embed_init = 1.0 / (params.vocab_size * params.hidden_size)
                W = tf.get_variable(
                    'embedding',
                    [params.vocab_size, params.hidden_size],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_embed_init
                    )
                )
            else:
                W = tf.get_variable(
                    'embedding',
                    initializer=W_pretrained
                )

            # Do an embedding lookup for each word in each sequence
            self.embeddings = tf.nn.embedding_lookup(W, x)

            # Create embedding prior matrix
            if not W_initializer is None:
                W_prior = tf.get_variable(
                    'embedding_prior',
                    initializer=W_initializer,
                    trainable=False
                )
                self.embeddings_prior = tf.nn.embedding_lookup(W_prior, x)
                W_prior_shape = W_initializer.shape

                # Create embedding prior projection matrix
                if params.projection:
                    max_embed_init = 1.0 / (W_prior_shape[1].value * params.hidden_size)
                    W_prior_proj = tf.get_variable(
                        'embedding_prior_projection',
                        [W_prior_shape[1].value, params.hidden_size],
                        initializer=tf.random_uniform_initializer(
                            maxval=max_embed_init
                        )
                    )
                    embeddings_prior_reshape = tf.reshape(self.embeddings_prior, [-1, W_prior_shape[1].value])
                    embeddings_prior_projected = tf.matmul(embeddings_prior_reshape, W_prior_proj)
                    self.embeddings_prior = tf.reshape(embeddings_prior_projected, [self.b_s[0], self.b_s[1], params.hidden_size])

                # Lambda multiplication
                if not self.lambda_embeddings < 0.0:
                    self.embeddings_prior = tf.scalar_mul(self.lambda_embeddings, self.embeddings_prior)

                self.embeddings = tf.add(self.embeddings, self.embeddings_prior)

        # Create bias vector of DocNADE
        bias = tf.get_variable(
            'bias',
            [params.hidden_size],
            initializer=tf.constant_initializer(0)
        )

        ################################## Deep DocNADE Parameters ##################################

        # List of deep projection matrices
        W_list = []

        # List of deep bias vectors
        bias_list = []

        if params.deep:
            in_size = params.hidden_size
            for index, size in enumerate(params.deep_hidden_sizes):
                out_size = size
                max_embed_init = 1.0 / (in_size * out_size)
                W_temp = tf.get_variable(
                    'embedding_' + str(index),
                    [in_size, out_size],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_embed_init
                    )
                )
                bias_temp = tf.get_variable(
                    'bias_' + str(index),
                    [out_size],
                    initializer=tf.constant_initializer(0)
                )
                W_list.append(W_temp)
                bias_list.append(bias_temp)
                in_size = out_size

        #############################################################################################

        # Compute the hidden layer inputs: each gets summed embeddings of
        # previous words in an autoregressive manner
        def sum_embeddings(previous, current):
            return previous + current

        h = tf.scan(sum_embeddings, tf.transpose(self.embeddings, [1, 2, 0]))
        h = tf.transpose(h, [2, 0, 1])

        # add initial zero vector to each sequence, will then generate the
        # first element using just the bias term
        h = tf.concat([
            tf.zeros([batch_size, 1, params.hidden_size], dtype=tf.float32), h
        ], axis=1)
        
        self.pre_act = h

        # Apply activation
        if params.activation == 'sigmoid':
            h = tf.sigmoid(h + bias)
        elif params.activation == 'tanh':
            h = tf.tanh(h + bias)
        elif params.activation == 'relu':
            h = tf.nn.relu(h + bias)
        else:
            print('Invalid value for activation: %s' % (params.activation))
            exit()
        self.aft_act = h

        # Extract final state for each sequence (document representation) in the batch
        indices = tf.stack([
            tf.range(batch_size),
            tf.to_int32(seq_lengths)
        ], axis=1)
        self.indices = indices
        self.h = tf.gather_nd(h, indices)

        h = h[:, :-1, :]
        h = tf.reshape(h, [-1, params.hidden_size])

        ################ Deep DocNADE forward propagation ###################

        if params.deep:
            for i in range(len(params.deep_hidden_sizes)):
                h = tf.nn.xw_plus_b(h, W_list[i], bias_list[i])
                self.h = tf.nn.xw_plus_b(self.h, W_list[i], bias_list[i])
                if params.activation == 'sigmoid':
                    h = tf.sigmoid(h)
                    self.h = tf.sigmoid(self.h)
                elif params.activation == 'tanh':
                    h = tf.tanh(h)
                    self.h = tf.tanh(self.h)
                else:
                    h = tf.nn.relu(h)
                    self.h = tf.nn.relu(self.h)

        ###################### Softmax logits ###############################

        if not params.num_samples:
            # Applying full softmax projection of word hidden vectors
            self.logits, _ = linear(h, params.vocab_size, scope='softmax', W_initializer=None)
            loss_function = None
        else:
            # Applying sampled softmax projection of word hidden vectors
            self.logits, _ = linear(h, params.num_samples, scope='softmax', W_initializer=None)
            if W_initializer is None:
                max_embed_init = 1.0 / (params.vocab_size * params.hidden_size)
                w_t = tf.get_variable(
                    "proj_w_t",
                    [params.vocab_size, params.num_samples],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_embed_init)
                )
            else:
                w_t = tf.get_variable(
                    "proj_w_t",
                    initializer=W_initializer
                )
            b = tf.get_variable("proj_b", [params.vocab_size])
            self.proj_w = tf.transpose(w_t)
            self.proj_b = b

            def sampled_loss(logits, labels):
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.nn.sampled_softmax_loss(
                    weights=local_w_t,
                    biases=local_b,
                    labels=labels,
                    inputs=local_inputs,
                    num_sampled=params.num_samples,
                    num_classes=params.vocab_size,
                    partition_strategy='div'
                )
            loss_function = sampled_loss

        """
        # Compute the loss. If using sampled softmax for training, use full
        # softmax for evaluation and validation
        if not params.num_samples:
            self.loss = masked_sequence_cross_entropy_loss(
                x,
                seq_lengths,
                self.logits
            )
        else:
            projected_logits = \
                tf.matmul(self.logits, self.proj_w) + self.proj_b
            self.loss = masked_sequence_cross_entropy_loss(
                x,
                seq_lengths,
                projected_logits
            )
        """

        # Calculates masked cross entropy loss
        self.loss_normed, self.labels, self.mask, self.loss_unnormed = masked_sequence_cross_entropy_loss(
            x,
            seq_lengths,
            self.logits,
            loss_function=loss_function,
            norm_by_seq_lengths=True
        )

        #Total DocNADE loss
        self.total_loss = self.loss_unnormed

        # Optimiser object
        step = tf.Variable(0, trainable=False)
        self.opt = gradients(
            opt=tf.train.AdamOptimizer(learning_rate=params.learning_rate),
            loss=self.total_loss,
            vars=tf.trainable_variables(),
            step=step
        )


class iDocNADE(object):
    '''
    Python class for creating iDocNADE object.
    '''
    def __init__(self, x, x_bw, y, seq_lengths, params,
                 W_initializer=None, lambda_embeddings=-1.0,
                 W_pretrained=None):
        '''
        x:           input of shape [batch_size X max document length] contains word indices
        x_bw:        input of shape [batch_size X max document length] contains word indices in reverse order of x
        y:           document labels
        seq_lengths: actual sequence lengths of documents in the padded matrix input x
        params:      training comfiguration parameters
        '''
        self.x = x
        self.x_bw = x_bw
        self.y = y
        self.seq_lengths = seq_lengths

        batch_size = tf.shape(x)[0]
        self.b_s = tf.shape(x)
        self.initializer_embeddings = W_initializer
        self.lambda_embeddings = lambda_embeddings

        # Create W (embedding matrix) of iDocNADE
        with tf.device('/cpu:0'):
            if W_pretrained is None:
                max_embed_init = 1.0 / (params.vocab_size * params.hidden_size)
                W = tf.get_variable(
                    'embedding',
                    [params.vocab_size, params.hidden_size],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_embed_init
                    )
                )
            else:
                W = tf.get_variable(
                    'embedding',
                    initializer=W_pretrained
                )

            # Do an embedding lookup for each word in each forward and backward sequence
            self.embeddings = tf.nn.embedding_lookup(W, x)
            self.embeddings_bw = tf.nn.embedding_lookup(W, x_bw)

            # Create embedding prior matrix
            if not W_initializer is None:
                W_prior = tf.get_variable(
                    'embedding_prior',
                    initializer=W_initializer,
                    trainable=False
                )
                self.embeddings_prior = tf.nn.embedding_lookup(W_prior, x)
                self.embeddings_prior_bw = tf.nn.embedding_lookup(W_prior, x_bw)
                W_prior_shape = W_initializer.shape

                # Create embedding prior projection matrix
                if params.projection:
                    max_embed_init = 1.0 / (W_prior_shape[1].value * params.hidden_size)
                    W_prior_proj = tf.get_variable(
                        'embedding_prior_projection',
                        [W_prior_shape[1].value, params.hidden_size],
                        initializer=tf.random_uniform_initializer(
                            maxval=max_embed_init
                        )
                    )
                    embeddings_prior_reshape = tf.reshape(self.embeddings_prior, [-1, W_prior_shape[1].value])
                    embeddings_prior_projected = tf.matmul(embeddings_prior_reshape, W_prior_proj)
                    self.embeddings_prior = tf.reshape(embeddings_prior_projected, [self.b_s[0], self.b_s[1], params.hidden_size])

                    embeddings_prior_bw_reshape = tf.reshape(self.embeddings_prior_bw, [-1, W_prior_shape[1].value])
                    embeddings_prior_bw_projected = tf.matmul(embeddings_prior_bw_reshape, W_prior_proj)
                    self.embeddings_prior_bw = tf.reshape(embeddings_prior_bw_projected, [self.b_s[0], self.b_s[1], params.hidden_size])

                # Lambda multiplication
                if not self.lambda_embeddings < 0.0:
                    self.embeddings_prior = tf.scalar_mul(self.lambda_embeddings, self.embeddings_prior)
                    self.embeddings_prior_bw = tf.scalar_mul(self.lambda_embeddings, self.embeddings_prior_bw)
                
                self.embeddings = tf.add(self.embeddings, self.embeddings_prior)
                self.embeddings_bw = tf.add(self.embeddings_bw, self.embeddings_prior_bw)

        # Create forward bias vector of iDocNADE
        bias = tf.get_variable(
            'bias',
            [params.hidden_size],
            initializer=tf.constant_initializer(0)
        )

        # Create backward bias vector of iDocNADE
        bias_bw = tf.get_variable(
            'bias_bw',
            [params.hidden_size],
            initializer=tf.constant_initializer(0)
        )

        ################################## Deep DocNADE Parameters ##################################

        # List of deep projection matrices
        W_list = []

        # List of deep forward bias vectors
        bias_list = []

        # List of deep backward bias vectors
        bias_bw_list = []

        if params.deep:
            in_size = params.hidden_size
            for index, size in enumerate(params.deep_hidden_sizes):
                out_size = size
                max_embed_init = 1.0 / (in_size * out_size)
                W_temp = tf.get_variable(
                    'embedding_' + str(index),
                    [in_size, out_size],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_embed_init
                    )
                )
                bias_temp = tf.get_variable(
                    'bias_' + str(index),
                    [out_size],
                    initializer=tf.constant_initializer(0)
                )
                bias_bw_temp = tf.get_variable(
                    'bias_bw_' + str(index),
                    [out_size],
                    initializer=tf.constant_initializer(0)
                )
                W_list.append(W_temp)
                bias_list.append(bias_temp)
                bias_bw_list.append(bias_bw_temp)
                in_size = out_size

        #############################################################################################

        # Compute the hidden layer inputs: each gets summed embeddings of
        # previous words
        def sum_embeddings(previous, current):
            return previous + current

        h = tf.scan(sum_embeddings, tf.transpose(self.embeddings, [1, 2, 0]))
        h_bw = tf.scan(sum_embeddings, tf.transpose(self.embeddings_bw, [1, 2, 0]))
        
        h = tf.transpose(h, [2, 0, 1])
        h_bw = tf.transpose(h_bw, [2, 0, 1])

        # add initial zero vector to each sequence, will then generate the
        # first element using just the bias term
        h = tf.concat([
            tf.zeros([batch_size, 1, params.hidden_size], dtype=tf.float32), h
        ], axis=1)

        h_bw = tf.concat([
            tf.zeros([batch_size, 1, params.hidden_size], dtype=tf.float32), h_bw
        ], axis=1)
        
        self.pre_act = h
        self.pre_act_bw = h_bw

        # Apply activation
        if params.activation == 'sigmoid':
            h = tf.sigmoid(h + bias)
            h_bw = tf.sigmoid(h_bw + bias_bw)
        elif params.activation == 'tanh':
            h = tf.tanh(h + bias)
            h_bw = tf.tanh(h_bw + bias_bw)
        elif params.activation == 'relu':
            h = tf.nn.relu(h + bias)
            h_bw = tf.nn.relu(h_bw + bias_bw)
        else:
            print('Invalid value for activation: %s' % (params.activation))
            exit()
        
        self.aft_act = h
        self.aft_act_bw = h_bw

        # Extract final state for each forward sequence (document representation) in the batch
        indices = tf.stack([
            tf.range(batch_size),
            tf.to_int32(seq_lengths)
        ], axis=1)

        # Extract final state for each backward sequence (document representation) in the batch
        indices_bw = tf.stack([
            tf.range(batch_size),
            tf.to_int32(seq_lengths)
        ], axis=1)

        self.indices = indices
        self.indices_bw = indices_bw

        self.h = tf.gather_nd(h, indices)
        self.h_bw = tf.gather_nd(h_bw, indices_bw)

        h = h[:, :-1, :]
        h_bw = h_bw[:, :-1, :]

        h = tf.reshape(h, [-1, params.hidden_size])
        h_bw = tf.reshape(h_bw, [-1, params.hidden_size])

        self.h_comb_concat = tf.concat([
            self.h, self.h_bw
        ], axis=1)
        self.h_comb_sum = tf.add(self.h, self.h_bw)

        ################ Deep network forward propagation ###################

        if params.deep:
            for i in range(len(params.deep_hidden_sizes)):
                h = tf.nn.xw_plus_b(h, W_list[i], bias_list[i])
                h_bw = tf.nn.xw_plus_b(h_bw, W_list[i], bias_bw_list[i])
                self.h = tf.nn.xw_plus_b(self.h, W_list[i], bias_list[i])
                self.h_bw = tf.nn.xw_plus_b(self.h_bw, W_list[i], bias_bw_list[i])
                if params.activation == 'sigmoid':
                    h = tf.sigmoid(h)
                    h_bw = tf.sigmoid(h_bw)
                    self.h = tf.sigmoid(self.h)
                    self.h_bw = tf.sigmoid(self.h_bw)
                elif params.activation == 'tanh':
                    h = tf.tanh(h)
                    h_bw = tf.tanh(h_bw)
                    self.h = tf.tanh(self.h)
                    self.h_bw = tf.tanh(self.h_bw)
                else:
                    h = tf.nn.relu(h)
                    h_bw = tf.nn.relu(h_bw)
                    self.h = tf.nn.relu(self.h)
                    self.h_bw = tf.nn.relu(self.h_bw)

        ###################### Softmax logits ###############################

        if not params.num_samples:
            # Applying full softmax projection of forward and backward word hidden vectors
            self.logits, self.logits_bw = linear(h, params.vocab_size, input_bw=h_bw, scope='softmax', W_initializer=None)
            loss_function = None
        else:
            # Applying sampled softmax projection of forward and backward word hidden vectors
            self.logits, self.logits_bw = linear(h, params.num_samples, input_bw=h_bw, scope='softmax', W_initializer=None)

            if W_initializer is None:
                max_embed_init = 1.0 / (params.vocab_size * params.hidden_size)
                w_t = tf.get_variable(
                    "proj_w_t",
                    [params.vocab_size, params.num_samples],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_embed_init)
                )
            else:
                w_t = tf.get_variable(
                    "proj_w_t",
                    initializer=W_initializer
                )
            b = tf.get_variable("proj_b", [params.vocab_size])
            self.proj_w = tf.transpose(w_t)
            self.proj_b = b

            def sampled_loss(logits, labels):
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.nn.sampled_softmax_loss(
                    weights=local_w_t,
                    biases=local_b,
                    labels=labels,
                    inputs=local_inputs,
                    num_sampled=params.num_samples,
                    num_classes=params.vocab_size,
                    partition_strategy='div'
                )
            loss_function = sampled_loss
        """
        # Compute the loss. If using sampled softmax for training, use full
        # softmax for evaluation and validation
        if not params.num_samples:
            self.loss = masked_sequence_cross_entropy_loss(
                x,
                seq_lengths,
                self.logits
            )
        else:
            projected_logits = \
                tf.matmul(self.logits, self.proj_w) + self.proj_b
            self.loss = masked_sequence_cross_entropy_loss(
                x,
                seq_lengths,
                projected_logits
            )
        """

        # Calculates masked cross entropy loss for forward network
        self.loss_normed, self.labels, self.mask, self.loss_unnormed = masked_sequence_cross_entropy_loss(
            x,
            seq_lengths,
            self.logits,
            loss_function=loss_function,
            norm_by_seq_lengths=True
        )

        # Calculates masked cross entropy loss for backward network
        self.loss_normed_bw, _, _, self.loss_unnormed_bw = masked_sequence_cross_entropy_loss(
            x_bw,
            seq_lengths,
            self.logits_bw,
            loss_function=loss_function,
            norm_by_seq_lengths=True
        )

        # Total iDocNADE (bidirectional DocNADE) loss
        self.total_loss = 0.5 * (self.loss_unnormed + self.loss_unnormed_bw)

        # Optimiser object
        step = tf.Variable(0, trainable=False)
        self.opt = gradients(
            opt=tf.train.AdamOptimizer(learning_rate=params.learning_rate),
            loss=self.total_loss,
            vars=tf.trainable_variables(),
            step=step
        )



class DocNADE_reload(object):
    '''
    Python class for reloading DocNADE object.
    '''
    def __init__(self, x, y, seq_lengths, params, W_initializer=None,
                 W_reload=None, W_prior_reload=None, W_prior_proj_reload=None, bias_reload=None, 
                 bias_bw_reload=None, V_reload=None, b_reload=None, b_bw_reload=None,
                 W_list_reload=None, bias_list_reload=None, lambda_embeddings=None):
        self.x = x
        self.y = y
        self.seq_lengths = seq_lengths

        batch_size = tf.shape(x)[0]
        self.b_s = tf.shape(x)

        self.V = V_reload
        self.b = b_reload
        self.W = W_reload

        self.lambda_embeddings = lambda_embeddings

        # Reload W (embedding matrix) of DocNADE
        W = tf.Variable(
            initial_value=W_reload,
            trainable=False
        )

        # Do an embedding lookup for each word in each sequence
        self.embeddings = tf.nn.embedding_lookup(W, x)

        # Reload embedding prior matrix
        if not W_initializer is None:
            W_prior = tf.Variable(
                initial_value=W_prior_reload,
                trainable=False
            )
            self.embeddings_prior = tf.nn.embedding_lookup(W_prior, x)
            W_prior_shape = W_initializer.shape

            # Reload embedding prior projection matrix
            if params['projection']:
                W_prior_proj = tf.Variable(
                    initial_value=W_prior_proj_reload,
                    trainable=False
                )
                embeddings_prior_reshape = tf.reshape(self.embeddings_prior, [-1, W_prior_shape[1].value])
                embeddings_prior_projected = tf.matmul(embeddings_prior_reshape, W_prior_proj)
                self.embeddings_prior = tf.reshape(embeddings_prior_projected, [self.b_s[0], self.b_s[1], params['hidden_size']])

            # Lambda multiplication
            if not self.lambda_embeddings < 0.0:
                self.embeddings_prior = tf.scalar_mul(self.lambda_embeddings, self.embeddings_prior)
            
            self.embeddings = tf.add(self.embeddings, self.embeddings_prior)

        # Reload bias vector of DocNADE
        bias = tf.Variable(
            initial_value=bias_reload,
            trainable=False
        )

        ########################## Deep DocNADE Parameters ##########################

        W_list = []
        bias_list = []

        if params['deep']:
            for index, size in enumerate(params['deep_hidden_sizes']):
                W_temp = tf.Variable(
                    initial_value=W_list_reload[index],
                    trainable=False
                )
                bias_temp = tf.Variable(
                    initial_value=bias_list_reload[index],
                    trainable=False
                )
                W_list.append(W_temp)
                bias_list.append(bias_temp)

        #############################################################################

        # Compute the hidden layer inputs: each gets summed embeddings of
        # previous words
        def sum_embeddings(previous, current):
            return previous + current

        h = tf.scan(sum_embeddings, tf.transpose(self.embeddings, [1, 2, 0]))
        h = tf.transpose(h, [2, 0, 1])

        # add initial zero vector to each sequence, will then generate the
        # first element using just the bias term
        h = tf.concat([
            tf.zeros([batch_size, 1, params['hidden_size']], dtype=tf.float32), h
        ], axis=1)
        self.pre_act = h

        # Apply activation
        if params['activation'] == 'sigmoid':
            h = tf.sigmoid(h + bias)
        elif params['activation'] == 'tanh':
            h = tf.tanh(h + bias)
        elif params['activation'] == 'relu':
            h = tf.nn.relu(h + bias)
        else:
            print('Invalid value for activation: %s' % (params['activation']))
            exit()
        self.aft_act = h

        # Extract final state for each sequence in the batch
        indices = tf.stack([
            tf.range(batch_size),
            tf.to_int32(seq_lengths)
        ], axis=1)
        self.indices = indices
        self.h = tf.gather_nd(h, indices)

        h = h[:, :-1, :]
        h = tf.reshape(h, [-1, params['hidden_size']])

        ################ Deep network forward propagation ###################

        if params['deep']:
            for i in range(len(params['deep_hidden_sizes'])):
                h = tf.nn.xw_plus_b(h, W_list[i], bias_list[i])
                self.h = tf.nn.xw_plus_b(self.h, W_list[i], bias_list[i])
                if params['activation'] == 'sigmoid':
                    h = tf.sigmoid(h)
                    self.h = tf.sigmoid(self.h)
                elif params['activation'] == 'tanh':
                    h = tf.tanh(h)
                    self.h = tf.tanh(self.h)
                else:
                    h = tf.nn.relu(h)
                    self.h = tf.nn.relu(self.h)

        ###################### Softmax logits ###############################

        self.logits, _ = linear_reload(h, params['vocab_size'], scope='softmax', W_initializer=None, 
                                        V_reload=V_reload, b_reload=b_reload)
        loss_function = None

        # Calculates masked cross entropy loss
        self.loss_normed, self.labels, self.mask, self.loss_unnormed = masked_sequence_cross_entropy_loss(
            x,
            seq_lengths,
            self.logits,
            loss_function=loss_function,
            norm_by_seq_lengths=True
        )


class iDocNADE_reload(object):
    '''
    Python class for reloading iDocNADE object.
    '''
    def __init__(self, x, x_bw, y, seq_lengths, params, W_initializer=None,
                 W_reload=None, W_prior_reload=None, W_prior_proj_reload=None, bias_reload=None, 
                 bias_bw_reload=None, V_reload=None, b_reload=None, b_bw_reload=None,
                 W_list_reload=[], bias_list_reload=[], bias_bw_list_reload=[], lambda_embeddings=None):
        self.x = x
        self.x_bw = x_bw
        self.y = y
        self.seq_lengths = seq_lengths

        batch_size = tf.shape(x)[0]
        self.b_s = tf.shape(x)

        self.V = V_reload
        self.b = b_reload
        self.W = W_reload

        self.lambda_embeddings = lambda_embeddings

        # Reload W (embedding matrix) of iDocNADE
        W = tf.Variable(
            initial_value=W_reload,
            trainable=False
        )

        # Do an embedding lookup for each word in each forward and backward sequence
        self.embeddings = tf.nn.embedding_lookup(W, x)
        self.embeddings_bw = tf.nn.embedding_lookup(W, x_bw)

        # Reload embedding prior matrix
        if not W_initializer is None:
            W_prior = tf.Variable(
                initial_value=W_prior_reload,
                trainable=False
            )
            self.embeddings_prior = tf.nn.embedding_lookup(W_prior, x)
            self.embeddings_prior_bw = tf.nn.embedding_lookup(W_prior, x_bw)
            W_prior_shape = W_initializer.shape

            # Reload embedding prior projection matrix
            if params['projection']:
                W_prior_proj = tf.Variable(
                    initial_value=W_prior_proj_reload,
                    trainable=False
                )
                embeddings_prior_reshape = tf.reshape(self.embeddings_prior, [-1, W_prior_shape[1].value])
                embeddings_prior_projected = tf.matmul(embeddings_prior_reshape, W_prior_proj)
                self.embeddings_prior = tf.reshape(embeddings_prior_projected, [self.b_s[0], self.b_s[1], params['hidden_size']])

                embeddings_prior_bw_reshape = tf.reshape(self.embeddings_prior_bw, [-1, W_prior_shape[1].value])
                embeddings_prior_bw_projected = tf.matmul(embeddings_prior_bw_reshape, W_prior_proj)
                self.embeddings_prior_bw = tf.reshape(embeddings_prior_bw_projected, [self.b_s[0], self.b_s[1], params['hidden_size']])

            # Lambda multiplication
            if not self.lambda_embeddings < 0.0:
                self.embeddings_prior = tf.scalar_mul(self.lambda_embeddings, self.embeddings_prior)
                self.embeddings_prior_bw = tf.scalar_mul(self.lambda_embeddings, self.embeddings_prior_bw)
            
            self.embeddings = tf.add(self.embeddings, self.embeddings_prior)
            self.embeddings_bw = tf.add(self.embeddings_bw, self.embeddings_prior_bw)

        # Reload forward bias vector of iDocNADE
        bias = tf.Variable(
            initial_value=bias_reload,
            trainable=False
        )

        # Reload backward bias vector of iDocNADE
        bias_bw = tf.Variable(
            initial_value=bias_bw_reload,
            trainable=False
        )

        ######################### Deep DocNADE Parameters ######################
        W_list = []
        bias_list = []
        bias_bw_list = []

        if params['deep']:
            for index, size in enumerate(params['deep_hidden_sizes']):
                W_temp = tf.Variable(
                    initial_value=W_list_reload[index],
                    trainable=False
                )
                bias_temp = tf.Variable(
                    initial_value=bias_list_reload[index],
                    trainable=False
                )
                bias_bw_temp = tf.Variable(
                    initial_value=bias_bw_list_reload[index],
                    trainable=False
                )
                W_list.append(W_temp)
                bias_list.append(bias_temp)
                bias_bw_list.append(bias_bw_temp)

        #########################################################################

        # Compute the hidden layer inputs: each gets summed embeddings of
        # previous words
        def sum_embeddings(previous, current):
            return previous + current

        h = tf.scan(sum_embeddings, tf.transpose(self.embeddings, [1, 2, 0]))
        h_bw = tf.scan(sum_embeddings, tf.transpose(self.embeddings_bw, [1, 2, 0]))
        
        h = tf.transpose(h, [2, 0, 1])
        h_bw = tf.transpose(h_bw, [2, 0, 1])

        # add initial zero vector to each sequence, will then generate the
        # first element using just the bias term
        h = tf.concat([
            tf.zeros([batch_size, 1, params['hidden_size']], dtype=tf.float32), h
        ], axis=1)

        h_bw = tf.concat([
            tf.zeros([batch_size, 1, params['hidden_size']], dtype=tf.float32), h_bw
        ], axis=1)
        
        self.pre_act = h
        self.pre_act_bw = h_bw

        # Apply activation
        if params['activation'] == 'sigmoid':
            h = tf.sigmoid(h + bias)
            h_bw = tf.sigmoid(h_bw + bias_bw)
        elif params['activation'] == 'tanh':
            h = tf.tanh(h + bias)
            h_bw = tf.tanh(h_bw + bias_bw)
        elif params['activation'] == 'relu':
            h = tf.nn.relu(h + bias)
            h_bw = tf.nn.relu(h_bw + bias_bw)
        else:
            print('Invalid value for activation: %s' % (params['activation']))
            exit()
        
        self.aft_act = h
        self.aft_act_bw = h_bw

        # Extract final state for each forward sequence (document representation) in the batch
        indices = tf.stack([
            tf.range(batch_size),
            tf.to_int32(seq_lengths)
        ], axis=1)

        # Extract final state for each backward sequence (document representation) in the batch
        indices_bw = tf.stack([
            tf.range(batch_size),
            tf.to_int32(seq_lengths)
        ], axis=1)

        self.indices = indices
        self.indices_bw = indices_bw

        self.h = tf.gather_nd(h, indices)
        self.h_bw = tf.gather_nd(h_bw, indices_bw)

        h = h[:, :-1, :]
        h_bw = h_bw[:, :-1, :]

        h = tf.reshape(h, [-1, params['hidden_size']])
        h_bw = tf.reshape(h_bw, [-1, params['hidden_size']])

        self.h_comb_concat = tf.concat([
            self.h, self.h_bw
        ], axis=1)
        self.h_comb_sum = tf.add(self.h, self.h_bw)

        ################ Deep network forward propagation ###################

        if params['deep']:
            for i in range(len(params['deep_hidden_sizes'])):
                h = tf.nn.xw_plus_b(h, W_list[i], bias_list[i])
                h_bw = tf.nn.xw_plus_b(h_bw, W_list[i], bias_bw_list[i])
                self.h = tf.nn.xw_plus_b(self.h, W_list[i], bias_list[i])
                self.h_bw = tf.nn.xw_plus_b(self.h_bw, W_list[i], bias_bw_list[i])
                if params['activation'] == 'sigmoid':
                    h = tf.sigmoid(h)
                    h_bw = tf.sigmoid(h_bw)
                    self.h = tf.sigmoid(self.h)
                    self.h_bw = tf.sigmoid(self.h_bw)
                elif params['activation'] == 'tanh':
                    h = tf.tanh(h)
                    h_bw = tf.tanh(h_bw)
                    self.h = tf.tanh(self.h)
                    self.h_bw = tf.tanh(self.h_bw)
                else:
                    h = tf.nn.relu(h)
                    h_bw = tf.nn.relu(h_bw)
                    self.h = tf.nn.relu(self.h)
                    self.h_bw = tf.nn.relu(self.h_bw)

        ###################### Softmax logits ###############################

        self.logits, self.logits_bw = linear_reload(h, params['vocab_size'], input_bw=h_bw, scope='softmax', W_initializer=None,
                                            V_reload=V_reload, b_reload=b_reload, b_bw_reload=b_bw_reload)
        loss_function = None

        # Calculates masked cross entropy loss for forward network
        self.loss_normed, self.labels, self.mask, self.loss_unnormed = masked_sequence_cross_entropy_loss(
            x,
            seq_lengths,
            self.logits,
            loss_function=loss_function,
            norm_by_seq_lengths=True
        )

        # Calculates masked cross entropy loss for backward network
        self.loss_normed_bw, _, _, self.loss_unnormed_bw = masked_sequence_cross_entropy_loss(
            x_bw,
            seq_lengths,
            self.logits_bw,
            loss_function=loss_function,
            norm_by_seq_lengths=True
        )