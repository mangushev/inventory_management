import os
import sys
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_addons as tfa

#tf.debugging.set_log_device_placement(True)

import numpy as np
np.set_printoptions(edgeitems=25, linewidth=10000, precision=12, suppress=True)

FLAGS = None

class Dense(tf.Module):
  def __init__(self, input_dim, output_size, activation=None, stddev=1.0):
    super(Dense, self).__init__()
    self.w = tf.Variable(
      tf.random.truncated_normal([input_dim, output_size], stddev=stddev), name='w')
    self.b = tf.Variable(tf.zeros([output_size]), name='b')
    self.activation = activation
  def __call__(self, x):
    y = tf.matmul(x, self.w) + self.b
    if (self.activation):
      y = self.activation(y)
    return y

#Policy network
class Actor(tf.Module):
  def __init__(self, num_features, num_actions, hidden_size, activation=tf.nn.relu, dropout_prob=0.1):
    super(Actor, self).__init__()
    self.layer1 = Dense(num_features, hidden_size, activation=None)
    self.layer2 = Dense(hidden_size, hidden_size, activation=None)
    self.layer3 = Dense(hidden_size, hidden_size, activation=None)
    self.layer4 = Dense(hidden_size, num_actions, activation=None)
    self.activation = activation
    self.dropout_prob = dropout_prob
  def __call__(self, state):
    #[I, P] --> [I]
    layer_output = self.layer1(state)
    #layer_output = tfa.layers.GroupNormalization(groups = 1)(layer_output) 
    layer_output = self.activation(layer_output)
    layer_output = tf.nn.dropout(layer_output, self.dropout_prob)

    layer_output = self.layer2(layer_output)
    #layer_output = tfa.layers.GroupNormalization(groups = 1)(layer_output) 
    layer_output = self.activation(layer_output)
    layer_output = tf.nn.dropout(layer_output, self.dropout_prob)

    layer_output = self.layer3(layer_output)
    #layer_output = tfa.layers.GroupNormalization(groups = 1)(layer_output) 
    layer_output = self.activation(layer_output)
    layer_output = tf.nn.dropout(layer_output, self.dropout_prob)

    layer_output = self.layer4(layer_output)
    #tf.print("layer_output:", tf.reduce_mean(layer_output, keepdims=False), layer_output[:10], output_stream=sys.stderr, summarize=-1)

    # 0 <= u <= 1 eq 3
    #layer_output = tf.nn.sigmoid(layer_output)
    return tf.nn.softmax(layer_output)
    #tf.print("sigmoid:", tf.reduce_mean(layer_output, keepdims=False), layer_output[:10], output_stream=sys.stderr, summarize=-1)

    #[I, 1] --> [I]
    #return tf.squeeze(layer_output, axis=-1, name='factor_squeeze')

#Value network
class Critic(tf.Module):
  def __init__(self, num_features, hidden_size, activation=tf.nn.relu, dropout_prob=0.1):
    super(Critic, self).__init__()
    self.layer1 = Dense(num_features, hidden_size, activation=None)
    self.layer2 = Dense(hidden_size, 1, activation=None)
    self.activation = activation
    self.dropout_prob = dropout_prob
  def __call__(self, state):
    #[I, P] --> [I]
    layer_output = self.layer1(state)
    layer_output = tfa.layers.GroupNormalization(groups = 1)(layer_output) 
    layer_output = self.activation(layer_output)
    layer_output = tf.nn.dropout(layer_output, self.dropout_prob)

    layer_output = self.layer2(layer_output)

    #[I, 1] --> [I]
    return tf.squeeze(layer_output, axis=-1, name='factor_squeeze')

def sales_parser(serialized_example):
  example = tf.io.parse_single_example(
    serialized_example,
    features={
      "sales": tf.io.FixedLenFeature([FLAGS.num_products], tf.float32)
    })

  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_float32(t)
      example[name] = t

  return example

def capacity_parser(serialized_example):
  example = tf.io.parse_single_example(
    serialized_example,
    features={
      "capacity": tf.io.FixedLenFeature([FLAGS.num_products], tf.float32)
    })

  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_float32(t)
      example[name] = t

  return example

def stock_parser(serialized_example):
  example = tf.io.parse_single_example(
    serialized_example,
    features={
      "stock": tf.io.FixedLenFeature([FLAGS.num_products], tf.float32)
    })

  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_float32(t)
      example[name] = t

  return example

def waste(x):
   return FLAGS.waste * x

def quantile(x, q):

  return np.quantile(x, q)

  x = tf.sort(x, direction='ASCENDING')
  pos = q * tf.cast(tf.size(x), tf.float32)
  floor_pos = tf.floor(pos)
  int_pos = tf.cast(floor_pos, tf.int32)
   
  v_diff = x[int_pos+1]-x[int_pos]
  p_diff = pos - floor_pos
   
  return x[int_pos]+v_diff*p_diff

def cross_entropy(p, q):
  return -tf.reduce_mean(tf.reduce_sum(p*tf.math.log(tf.math.maximum(1e-15, q)), axis=1))

class Env(tf.Module):
  def __init__(self, num_features, hidden_size, activation=tf.nn.relu, dropout_prob=0.1):
    super(Critic, self).__init__()
    self.layer1 = Dense(num_features, hidden_size, activation=None)
    self.layer2 = Dense(hidden_size, 1, activation=None)
    self.activation = activation
    self.dropout_prob = dropout_prob
  def __call__(self, u):
    #[I, P] --> [I]
    layer_output = self.layer1(state)
    layer_output = tfa.layers.GroupNormalization(groups = 1)(layer_output) 
    layer_output = self.activation(layer_output)
    layer_output = tf.nn.dropout(layer_output, self.dropout_prob)

    layer_output = self.layer2(layer_output)

    #[I, 1] --> [I]
    return tf.squeeze(layer_output, axis=-1, name='factor_squeeze')

def predict():
  sales_dataset = tf.data.TFRecordDataset(FLAGS.predict_file)
  capacity_dataset = tf.data.TFRecordDataset(FLAGS.capacity_file)
  stock_dataset = tf.data.TFRecordDataset(FLAGS.stock_file)

  parsed_capacity_dataset = capacity_dataset.map(capacity_parser)
  capacity = next(iter(parsed_capacity_dataset))['capacity']

  parsed_dataset = sales_dataset.map(sales_parser)

  parsed_stock_dataset = stock_dataset.map(stock_parser)
  x = next(iter(parsed_stock_dataset))['stock']
  #x = tf.divide(next(iter(parsed_stock_dataset))['stock'], capacity)

  actor = Actor(FLAGS.num_features, FLAGS.num_actions, FLAGS.hidden_size, activation=tf.nn.relu, dropout_prob=FLAGS.dropout_prob)

  checkpoint = tf.train.Checkpoint(actor=actor)
  checkpoint.restore(tf.train.latest_checkpoint(FLAGS.output_dir)).expect_partial()

  with tf.io.gfile.GFile(FLAGS.output_file, "w") as writer:
    for sales_record in parsed_dataset:
      
      sales = tf.divide(sales_record['sales'], capacity)

      q = waste(x)

      s = tf.transpose(tf.stack([x, sales, q], axis=0), perm=[1, 0])

      policy_probs = actor(s)
      policy_mask = tf.one_hot(tf.math.argmax(policy_probs, axis=-1), FLAGS.num_actions)
      action_space = tf.tile([[0, 0.005, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.03, 0.04, 0.08, 0.12, 0.2, 0.5, 1]], [FLAGS.num_products, 1])
      u = tf.boolean_mask(action_space, policy_mask)

      overstock = tf.math.maximum(0, (x + u) - 1)

      x_u = tf.math.minimum(1, x + u)

      stockout = tf.math.minimum(0, x_u - sales)

      writer.write("stock:" + ','.join(  list(map(str,   x.numpy()    ))    ) + "\n")
      writer.write("action:" + ','.join(  list(map(str,   u.numpy()    ))    ) + "\n")
      writer.write("overstock:" + ','.join(  list(map(str,   overstock.numpy()    ))    ) + "\n")
      writer.write("sales:" + ','.join(  list(map(str,   sales.numpy()    ))    ) + "\n")
      writer.write("stockout:" + ','.join(  list(map(str,   stockout.numpy()    ))    ) + "\n")
      writer.write("capacity:" + ','.join(  list(map(str,   (capacity/capacity).numpy()    ))    ) + "\n")

      x = tf.math.maximum(0, x_u - sales)

def train():
  #   LEGEND:
  #   p - number of products
  #   f - number of features
  #   t - number of timesteps in an episode
  #   n - number of actions
  #   ep - experience collection episodes

  #sales for FLAGS.num_timesteps time periods for NUM_PRODUCTS products. Sales for period [t, t+1]. so index t=0, sales from 0 until 1 
  sales_dataset = tf.data.TFRecordDataset(FLAGS.train_file).window(FLAGS.batch_size, shift=FLAGS.batch_size-1, drop_remainder=False)

  capacity_dataset = tf.data.TFRecordDataset(FLAGS.capacity_file) #, buffer_size=FLAGS.dataset_reader_buffer_size)
  parsed_capacity_dataset = capacity_dataset.map(capacity_parser)
  capacity = next(iter(parsed_capacity_dataset))['capacity']

  actor_optimizer = tf.optimizers.Adam(FLAGS.actor_learning_rate)
  critic_optimizer = tf.optimizers.Adam(FLAGS.critic_learning_rate)

  #Policy and Value networks with random weights 
  actor = Actor(FLAGS.num_features, FLAGS.num_actions, FLAGS.hidden_size, activation=tf.nn.relu, dropout_prob=FLAGS.dropout_prob)
  critic = Critic(FLAGS.num_features, FLAGS.hidden_size, activation=tf.nn.relu, dropout_prob=FLAGS.dropout_prob)

  #Counter
  global_step = tf.Variable(0)

  checkpoint_prefix = os.path.join(FLAGS.output_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(critic_optimizer=critic_optimizer, actor_optimizer=actor_optimizer, critic=critic, actor=actor, step=global_step)
  status = checkpoint.restore(tf.train.latest_checkpoint(FLAGS.output_dir))

  #standard deviation
  sigma = tf.constant(0.1)

  for episode in range(FLAGS.train_episodes):
    #random initial inventory
    # 0 <= x <= 1: eq 2
    x = tf.random.uniform(shape=[FLAGS.num_products], minval=0, maxval=1, dtype=tf.dtypes.float32)
    #waste 10% of grocery inventory at the begining of the day. This is q-hat estimate!
    # q-hat: estimate of waste
    q = waste(x)

    #tf.print ("start:", x, output_stream=sys.stderr, summarize=-1)

    for batch_dataset in sales_dataset:
      with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        experience_step = tf.constant(0)
        experience_s = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_products, FLAGS.num_features]), name="experience_s")
        experience_u = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_products]), name="experience_u")
        experience_p = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_products, FLAGS.num_actions]), name="experience_p")
        experience_i = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.int64, element_shape=tf.TensorShape([FLAGS.num_products]), name="experience_i")
        experience_pu = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_products]), name="experience_pu")
        experience_overstock = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_products]), name="experience_overstock")
        experience_s_prime = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_products, FLAGS.num_features]), name="experience_s_prime")
        experience_r = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_products]), name="experience_r_prime")
        experience_z = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_products]), name="experience_z")
        experience_q = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_products]), name="experience_q")
        experience_quan = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_products]), name="experience_quan")

        batch_iterator = batch_dataset.map(sales_parser)

        sales = tf.divide(next(iter(batch_iterator))['sales'], capacity)

        #state is starting inventory and forecast sales during this period
        #(p), (p) --> (f, p) --> (p, f)
        s = tf.transpose(tf.stack([x, sales, q], axis=0), perm=[1, 0])

        #tf.print("x:", x, output_stream=sys.stderr, summarize=-1)
        #tf.print("sales:", sales, output_stream=sys.stderr, summarize=-1)

        #(p, f) --> (p, n)
        policy_probs = actor(s)

        for item in batch_iterator:
          sales_prime = tf.divide(item['sales'], capacity)

          #(p, n) --> (p)
          policy_index = tf.squeeze(tf.random.categorical(tf.math.log(policy_probs), 1))
          
          #(p) --> (p, n)
          policy_mask = tf.one_hot(policy_index, FLAGS.num_actions)

          #(p, n), (p, n) --> (p)
          policy_selected = tf.boolean_mask(policy_probs, policy_mask)
     
          #tf.print("mask:", mask, output_stream=sys.stderr, summarize=-1)

          action_space = tf.tile([[0, 0.005, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.03, 0.04, 0.08, 0.12, 0.2, 0.5, 1]], [FLAGS.num_products, 1])

          #(p, n), (p, n) --> (p)
          u = tf.boolean_mask(action_space, policy_mask)

          #tf.print("action:", u, output_stream=sys.stderr, summarize=-1)

          overstock = tf.math.maximum(0, (x + u) - 1)

          # 0 <= x + u <= 1: eq 4 
          #(p) + (p) --> (p)
          x_u = tf.math.minimum(1, x + u)

          # 0 <= x <= 1: eq 7
          #(p) - (p) --> (p)
          x_prime = tf.math.maximum(0, x_u - sales)

          #tf.print("x_prime:", x_prime, output_stream=sys.stderr, summarize=-1)
        
          #waste 10% of grocery inventory at the begining of the day. This is q-hat estimate!
          # q-hat: estimate of waste
          q_prime = waste(x_prime)

          #(p), (p) --> (f, p) --> (p, f)
          s_prime = tf.transpose(tf.stack([x_prime, sales_prime, q_prime], axis=0), perm=[1, 0])

          z = tf.cast(x < FLAGS.zero_inventory, tf.float32)

          quan = tf.repeat(tf.cast(quantile(x, 0.95) - quantile(x, 0.05), tf.float32), FLAGS.num_products)

          #(p), (p), (p), (p) --> (p)
          r = tf.cast(1 - z - overstock - q - quan, tf.float32)

          #tf.print("rewards:", global_step, tf.reduce_mean(r, keepdims=False), output_stream=sys.stderr, summarize=-1)

          experience_s = experience_s.write(experience_step, s)
          experience_u = experience_u.write(experience_step, u)
          experience_p = experience_p.write(experience_step, policy_probs)
          experience_i = experience_i.write(experience_step, policy_index)
          experience_pu = experience_pu.write(experience_step, policy_selected)
          experience_overstock = experience_overstock.write(experience_step, overstock)
          experience_s_prime = experience_s_prime.write(experience_step, s_prime)
          experience_r = experience_r.write(experience_step, r)
          experience_z = experience_z.write(experience_step, z)
          experience_q = experience_q.write(experience_step, q)
          experience_quan = experience_quan.write(experience_step, quan)

          #(p, f) --> (p, n)
          policy_probs = actor(s_prime)

          x = x_prime
          q = q_prime
          s = s_prime
          sales = sales_prime

          experience_step = experience_step + 1

        #(t, p, f) --> (t*p, f)
        s_batch = tf.reshape(experience_s.stack()[:experience_step, :, :], [-1, FLAGS.num_features])
        x_batch = tf.reshape(experience_s.stack()[:experience_step, :, 0], [-1])
        sal_bat = tf.reshape(experience_s.stack()[:experience_step, :, 1], [-1])
        u_batch = tf.reshape(experience_u.stack()[:experience_step, :], [-1])
        p_batch = tf.reshape(experience_p.stack()[:experience_step, :], [-1, FLAGS.num_actions])
        i_batch = tf.reshape(experience_i.stack()[:experience_step, :], [-1])
        pu_batch = tf.reshape(experience_pu.stack()[:experience_step, :], [-1])
        overstock_batch = tf.reshape(experience_overstock.stack()[:experience_step, :], [-1])
        s_prime_batch = tf.reshape(experience_s_prime.stack()[:experience_step, :, :], [-1, FLAGS.num_features])
        r_batch = tf.reshape(experience_r.stack()[:experience_step, :], [-1])
        z_batch = tf.reshape(experience_z.stack()[:experience_step, :], [-1])
        q_batch = tf.reshape(experience_q.stack()[:experience_step, :], [-1])
        quan_batch = tf.reshape(experience_quan.stack()[:experience_step, :], [-1])

        tf.print("rewards:", global_step, experience_step, tf.reduce_mean(r_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)
        tf.print("stockouts:", global_step, experience_step, tf.reduce_mean(z_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)
        tf.print("waste:", global_step, experience_step, tf.reduce_mean(q_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)
        tf.print("quantile:", global_step, experience_step, tf.reduce_mean(quan_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)

        tf.print("x    :", global_step, experience_step, tf.reduce_mean(x_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)
        tf.print("u    :", global_step, experience_step, tf.reduce_mean(u_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)
        tf.print("p    :", global_step, experience_step, tf.reduce_mean(p_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)
        tf.print("pu   :", global_step, experience_step, tf.reduce_mean(pu_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)
        tf.print("o    :", global_step, experience_step, tf.reduce_mean(overstock_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)
        tf.print("sales:", global_step, experience_step, tf.reduce_mean(sal_bat, keepdims=False), output_stream=sys.stderr, summarize=-1)

        #tf.print("z_batch:", experience_step, z_batch, output_stream=sys.stderr, summarize=-1)
        #tf.print("q_batch:", experience_step, q_batch, output_stream=sys.stderr, summarize=-1)
        #tf.print("quan_batch:", experience_step, quan_batch, output_stream=sys.stderr, summarize=-1)
        #tf.print("sales_batch:", experience_step, sal_bat, output_stream=sys.stderr, summarize=-1)

        #(t*p, f) --> (t*p)
        v = critic(s_batch)

        #(t*p, f) --> (t*p)
        v_prime = critic(s_prime_batch)

        y = r_batch + FLAGS.gamma*v_prime

        #(t*p, t*p, t*p) --> (t*p)
        delta = y - v
        tf.print("delta:", global_step, tf.reduce_mean(delta, keepdims=False), output_stream=sys.stderr, summarize=-1)

        #(t*p) --> (1)
        critic_loss = 0.5*tf.reduce_mean(tf.math.square(delta), keepdims=False)
        tf.print("critic loss:", global_step, critic_loss, output_stream=sys.stderr, summarize=-1)

        if global_step == 0:
          tf.print("p_old == p_batch:", output_stream=sys.stderr, summarize=-1)
          pu_old = pu_batch

        #(t*p, n), (t*p, n) --> (t*p) --> (1)
        entropy_p = cross_entropy(p_batch, p_batch)
        tf.print("entropy adjusted:", global_step, FLAGS.entropy_coefficient*entropy_p, output_stream=sys.stderr, summarize=-1)

        if FLAGS.algorithm == 'A2C':
          #(t*p), (t*p), (1) --> (1), (1) --> (1)
          actor_loss = -tf.reduce_mean(tf.math.log(tf.math.maximum(1e-15, pu_batch))*tf.stop_gradient(delta), keepdims=False) - FLAGS.entropy_coefficient*entropy_p

          #(t*p) --> (t,p) --> (p) --> (1)
          #actor_loss = -tf.reduce_mean(tf.reduce_mean(tf.reshape(tf.math.log(tf.math.maximum(1e-15, pu_batch))*delta, [-1, FLAGS.num_products]), axis=0)) - FLAGS.entropy_coefficient*entropy_p
        elif FLAGS.algorithm == 'A2C_mod':
          #(t*p), ... --> (t*p,n)
          ix_batch = tf.tile(tf.reshape(i_batch, [-1, 1]), [1, FLAGS.num_actions])

          #(t*p,n) --> (t*p,n)
          p_new = tf.nn.softmax(tf.math.log(tf.math.maximum(1e-15, p_batch)) + tf.reshape(delta, [-1, 1]) / tf.cast(tf.math.abs(ix_batch - tf.cast(tf.range(FLAGS.num_actions), tf.int64)) + 1, tf.float32))
          #(t*p,n), ... --> (t*p)
          #per_timestep_actor_loss = tf.reduce_mean(tf.math.squared_difference(p_batch, p_new), axis=-1)
          #(t*p), ... --> (1)
          actor_loss = tf.reduce_mean(per_timestep_actor_loss, axis=-1)
        elif FLAGS.algorithm == 'PPO':
          r = pu_batch/pu_old

          #(t*p,), (t*p) --> (1)
          actor_loss = -tf.reduce_mean(tf.math.minimum(r*delta,tf.clip_by_value(r,1-0.2,1+0.2)*delta), keepdims=False) - FLAGS.entropy_coefficient*entropy_p

        tf.print("actor loss:", global_step, actor_loss, output_stream=sys.stderr, summarize=-1)

        pu_old = pu_batch

        global_step.assign_add(1)

      actor_gradients = actor_tape.gradient(actor_loss, actor.variables)
      #tf.print("actor grads:", global_step, actor_gradients, output_stream=sys.stderr, summarize=-1)

      critic_gradients = critic_tape.gradient(critic_loss, critic.variables)
      #tf.print("critic grads:", global_step, critic_gradients, output_stream=sys.stderr, summarize=-1)

      actor_optimizer.apply_gradients(zip(actor_gradients, actor.variables))
      critic_optimizer.apply_gradients(zip(critic_gradients, critic.variables))

    if (episode + 1) % 10 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

  ##tf.print ("ending:", x, output_stream=sys.stderr, summarize=-1)

  tf.print ("episode:", episode, global_step, output_stream=sys.stderr, summarize=-1)

def main():  
  if FLAGS.action == 'TRAIN':
    train()
  elif FLAGS.action == 'PREDICT':
    predict()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='checkpoints',
            help='Model directrory in google storage.')
    parser.add_argument('--train_file', type=str, default='data/train.tfrecords',
            help='Train file location in google storage.')
    parser.add_argument('--capacity_file', type=str, default='data/capacity.tfrecords',
            help='Shelf capacity file location in google storage.')
    parser.add_argument('--stock_file', type=str, default='data/stock.tfrecords',
            help='Stock values in prediction mode. It is random during the training.')
    parser.add_argument('--predict_file', type=str, default='data/test.tfrecords',
            help='Predict/Test file location in google storage.')
    parser.add_argument('--output_file', type=str, default='./output.csv',
            help='Prediction output.')
    parser.add_argument('--dropout_prob', type=float, default=0.1,
            help='This used for all dropouts.')
    parser.add_argument('--train_episodes', type=int, default=1000,
            help='How many times to run scenarious.')
    parser.add_argument('--num_products', type=int, default=100,
            help='How many productse. This is a subset of all products. They are some of grocery products.')
    parser.add_argument('--num_timesteps', type=int, default=1000,
            help='How many timesteps in an episode.')
    parser.add_argument('--num_features', type=int, default=3,
            help='How many features in Critic/Actor network.')
    parser.add_argument('--num_actions', type=int, default=14,
            help='How many actions for store replenishment.')
    parser.add_argument('--hidden_size', type=int, default=32,
            help='Actor and Critic layers hidden size.')
    parser.add_argument('--entropy_coefficient', type=float, default=0.001,
            help='Applied to entropy regularizing value for actor loss.')
    parser.add_argument('--gamma', type=float, default=0.99,
            help='Discount in future rewards.')
    parser.add_argument('--algorithm', default='A2C', choices=['A2C','A2C_mod','PPO'],
            help='Learning algorithm for critic and actor.')
    parser.add_argument('--waste', type=float, default=0.025,
            help='Waste of store stock for time period.')
    parser.add_argument('--num_experience_episodes', type=int, default=5,
            help='How many episodes to collect experience before starting training.')
    parser.add_argument('--num_training_epochs', type=int, default=40,
            help='How many epochs to train from experience buffer.')
    parser.add_argument('--actor_learning_rate', type=float, default=0.001,
            help='Optimizer learning rate for Actor.')
    parser.add_argument('--critic_learning_rate', type=float, default=0.001,
            help='Optimizer learning rate for Critic.')
    parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
    parser.add_argument('--zero_inventory', type=float, default=1e-5,
            help='Consider as zero inventory if less than that.')
    parser.add_argument('--batch_size', type=int, default=32,
            help='Batch size.')
    parser.add_argument('--action', default='PREDICT', choices=['TRAIN','EVALUATE','PREDICT'],
            help='An action to execure.')

    FLAGS, unparsed = parser.parse_known_args()



    main()
