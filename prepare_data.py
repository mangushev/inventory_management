
#TODO: 

import tensorflow as tf
import os
import argparse
import sys
import random
import math
import logging
import operator
import itertools
import datetime
import numpy as np
import pandas as pd
from csv import reader
from random import randrange

FLAGS = None

#FORMAT = '%(asctime)s %(levelname)s %(message)s'
#logging.basicConfig(format=FORMAT)
#logger = logging.getLogger('tensorflow')

logger = logging.getLogger('tensorflow')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.removeHandler(logger.handlers[0])
logger.propagate = False

def sales_example(sales):
  record = {
        'sales': tf.train.Feature(float_list=tf.train.FloatList(value=sales))
  }

  return tf.train.Example(features=tf.train.Features(feature=record))

def capacity_example(capacity):
  record = {
        'capacity': tf.train.Feature(float_list=tf.train.FloatList(value=capacity))
  }

  return tf.train.Example(features=tf.train.Features(feature=record))

def stock_example(stock):
  record = {
        'stock': tf.train.Feature(float_list=tf.train.FloatList(value=stock))
  }

  return tf.train.Example(features=tf.train.Features(feature=record))

#https://stackoverflow.com/questions/553303/generate-a-random-date-between-two-other-dates
def random_date(start, end):
  return start + datetime.timedelta(
    seconds=random.randint(0, int((end - start).total_seconds())),
  )

def create_records(number_of_products, start_date, end_date, start_time_period, middle_time_period, end_time_period, orders_file, products_file, departments_file, order_products_prior_file, order_products_train_file, train_tfrecords_file, test_tfrecords_file, capacity_tfrecords_file, stock_tfrecords_file):

  stock = np.random.uniform(low=0.0, high=1.0, size=(FLAGS.number_of_products))
  with tf.io.TFRecordWriter(stock_tfrecords_file) as writer:
    logger.debug ("stock: {}".format(stock))
    tf_example = stock_example(stock)
    writer.write(tf_example.SerializeToString())

  with open(orders_file, 'r') as f:
    csv_reader = reader(f)
    next(csv_reader)
    orders_list = list(map(tuple, csv_reader))

  sorted_orders = sorted(orders_list, key = lambda x: (int(x[1]), int(x[3])))

  dated_orders = []

  i = 0
  for k, g in itertools.groupby(sorted_orders, lambda x : int(x[1])):
    item = next(g)
    order_date = random_date(start_date, end_date)
    while order_date.weekday() != int(item[4]):
      order_date = order_date + datetime.timedelta(days=1)
        
    start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
    end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time())

    order_date = datetime.datetime(order_date.year, order_date.month, order_date.day, int(item[5]), 0, 0) 
    time_period = int((order_date - start_date).total_seconds() / (60*60*6))
    dated_orders.append((int(item[0]), int(item[1]), int(item[4]), order_date, time_period))
    
    for item in g:
      order_date = order_date + datetime.timedelta(days=int(float(item[6])))
      order_date = datetime.datetime(order_date.year, order_date.month, order_date.day, int(item[5]), 0, 0)
      time_period = int((order_date - start_date).total_seconds() / (60*60*6))
      dated_orders.append((int(item[0]), int(item[1]), int(item[4]), order_date, time_period))

  orders = pd.DataFrame(dated_orders, columns =['order_id', 'user_id', 'order_dow', 'order_date', 'time_period'])

  products = pd.read_csv(products_file)
  departments = pd.read_csv(departments_file)
  prior_order = pd.read_csv("data/order_products__prior.csv")
  train_order = pd.read_csv("data/order_products__train.csv")
  #aisles = pd.read_csv("data/aisles.csv")

  ntop = int(FLAGS.top_products*products['product_id'].count())

  all_ordered_products = pd.concat([prior_order, train_order], axis=0)[["order_id", "product_id"]]

  largest = all_ordered_products[['product_id']].groupby(['product_id']).size().nlargest(ntop).to_frame()
  largest.reset_index(inplace=True)

  products_largest = pd.merge(largest, products, how="left", on="product_id")[['product_id', 'product_name', 'aisle_id', 'department_id']]

  products_departments = pd.merge(products_largest, departments, how="left", on="department_id")

  products_departments = products_departments[products_departments["department"].isin(["frozen", "bakery", "produce", "beverages", "dry goods pasta", "meat seafood", "pantry", "breakfast", "canned goods", "dairy eggs", "snacks", "deli"])]

  products_departments_list = products_departments.values.tolist()

  products_subset=set()
  while len(products_subset) < number_of_products:
    products_subset.add((random.randint(0,len(products_departments_list))))

  selected_products_departments_list = [products_departments_list[i] for i in products_subset]
  selected_products_list = [products_departments_list[i][0] for i in products_subset]

  for p, product_id in enumerate(selected_products_list):
    logger.info ("{} {}".format(p, product_id))

  selected_products_departments = pd.DataFrame(selected_products_departments_list, columns =['product_id', 'product_name', 'aisle_id', 'department_id', 'department'])

  all_ordered_products_quantity_list = []
  for item in all_ordered_products.itertuples():
    all_ordered_products_quantity_list.append((item[1], item[2], 1))
    #all_ordered_products_quantity_list.append((item[1], item[2], random.randint(1, 6)))

  all_ordered_products_quantity = pd.DataFrame(all_ordered_products_quantity_list, columns =["order_id", "product_id", 'quantity'])

  order_product_departments = pd.merge(selected_products_departments, all_ordered_products_quantity, how="left", on="product_id")
  order_product_departments_dates = pd.merge(order_product_departments, orders, how="left", on="order_id")

  grocery = order_product_departments_dates[["order_id", "product_id", "product_name", "order_date", "time_period", 'quantity']]

  shelf_capacity = ((grocery.groupby(['product_id'])['quantity'].sum()/grocery['time_period'].nunique())*4*3).to_frame()
  shelf_capacity.reset_index(inplace=True)

  with tf.io.TFRecordWriter(capacity_tfrecords_file) as writer:
    capacity = []
    for p, product_id in enumerate(selected_products_list):
      capacity.append(math.ceil(shelf_capacity[shelf_capacity['product_id'] == product_id]['quantity'].values[0]))

    logger.debug ("capacity: {}".format(capacity))
    tf_example = capacity_example(np.array(capacity, dtype=np.float32))
    writer.write(tf_example.SerializeToString())

  counter = 0
  with tf.io.TFRecordWriter(train_tfrecords_file) as writer:
    for t in range(start_time_period, middle_time_period):
      sales = []
      for p, product_id in enumerate(selected_products_list):
        sales.append(grocery[(grocery['time_period'] == t) & (grocery['product_id'] == product_id)]['quantity'].sum())
 
      logger.debug ("pediod {}: {}".format(t, sales))
      tf_example = sales_example(np.array(sales, dtype=np.float32))
      writer.write(tf_example.SerializeToString())
      counter = counter + 1

  logger.info ("created {} train sales records".format(counter))

  if end_time_period == -1:
    end_time_period = grocery['time_period'].max()

  counter = 0
  with tf.io.TFRecordWriter(test_tfrecords_file) as writer:
    for t in range(middle_time_period, end_time_period+1):
      sales = []
      for p, product_id in enumerate(selected_products_list):
        sales.append(grocery[(grocery['time_period'] == t) & (grocery['product_id'] == product_id)]['quantity'].sum())
 
      logger.debug ("pediod {}: {}".format(t, sales))
      tf_example = sales_example(np.array(sales, dtype=np.float32))
      writer.write(tf_example.SerializeToString())
      counter = counter + 1

  logger.info ("created {} test sales records".format(counter))

def main():
  create_records(FLAGS.number_of_products, FLAGS.start_date, FLAGS.end_date, FLAGS.start_time_period, FLAGS.middle_time_period, FLAGS.end_time_period, FLAGS.orders_file, FLAGS.products_file, FLAGS.departments_file, FLAGS.order_products_prior_file, FLAGS.order_products_train_file, FLAGS.train_tfrecords_file, FLAGS.test_tfrecords_file, FLAGS.capacity_tfrecords_file, FLAGS.stock_tfrecords_file)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--number_of_products', type=int, default=100,
            help='Subset of products from whole ~50k dataset.')
  parser.add_argument('--top_products', type=float, default=0.2,
            help='Top percentage of products to consider, so shelf capacity equal to 3-days of sales will have a reasonable number > 1.')
  parser.add_argument('--start_date', type=datetime.date.fromisoformat, default='2017-01-01',
            help='Start date random range to create timestampts.')
  parser.add_argument('--end_date', type=datetime.date.fromisoformat, default='2017-01-06',
            help='End date random range to create timestampts.')
  parser.add_argument('--start_time_period', type=int, default=0,
            help='Start timestep for train dataset.')
  parser.add_argument('--middle_time_period', type=int, default=1000,
            help='End timestep for train dataset and this is the first timestep for test dataset.')
  parser.add_argument('--end_time_period', type=int, default=-1,
            help='Last timestep for test dataset. If -1 than until the end of data.')
  parser.add_argument('--orders_file', type=str, default='data/orders.csv',
            help='orders file location.')
  parser.add_argument('--products_file', type=str, default='data/products.csv',
            help='products file location.')
  parser.add_argument('--departments_file', type=str, default='data/departments.csv',
            help='departments file location.')
  parser.add_argument('--order_products_prior_file', type=str, default='data/order_products__prior.csv',
            help='order_products_prior file location.')
  parser.add_argument('--order_products_train_file', type=str, default='data/order_products__train.csv',
            help='order_products_train file location.')
  parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
  parser.add_argument('--train_tfrecords_file', type=str, default='data/train.tfrecords',
            help='train sales tfrecords output file')
  parser.add_argument('--test_tfrecords_file', type=str, default='data/test.tfrecords',
            help='test sales tfrecords output file')
  parser.add_argument('--capacity_tfrecords_file', type=str, default='data/capacity.tfrecords',
            help='shelf capacity tfrecords output file, train or test')
  parser.add_argument('--stock_tfrecords_file', type=str, default='data/stock.tfrecords',
            help='Stock data for each product for predict mode.')

  FLAGS, unparsed = parser.parse_known_args()

  logger.setLevel(FLAGS.logging)

  logger.debug ("Running with parameters: {}".format(FLAGS))

  main()
