# inventory_management

This code implements few papers about supply chain from the same authors

1. Scalable multi-product inventory control with lead time constraints using reinforcement learning

https://link.springer.com/article/10.1007/s00521-021-06129-w

2. Using Reinforcement Learning for a Large Variable-Dimensional Inventory Management Problem

https://ala2020.vub.ac.be/

https://ala2020.vub.ac.be/papers/ALA2020_paper_5.pdf

3. Reinforcement Learning for Multi-Product Multi-Node Inventory Management in Supply Chains

https://arxiv.org/abs/2006.04037

There are variety of papers on reinforcement learning and university cources

- A2C algorithm
- PPO algorithm

- Stanford RL cources on youtube
- University of berkeley on youtube

Data: I used the same data as in the papers.

Description:

Supply chain can have many objectives to achieve optimal metrics: lowest possible inventory level without having stockouts. This reduces amount of thrown away items. Another objective is to make quantity of items across all products as even as possible. Other problems with solution in the paper is addressing both stores and warehouses and different lead times. I have not coded this yet. Beside this, I saw in other papers could be some optimal prices in the store.   

This is reinforcement learnig problem and it is implemented in a actor critic style with separate networks. I coded A2C, A2C-mod as suggested in the paper and PPO. Actor is a policy network and prediction uses just it.

Notes:

- Only A2C with 1024 batch size works for now
- I have not implemented any baseline supply chain algorith to compare performance

Training steps:

1. Download Instacart dataset

this just what instacart asks to put. 

“The Instacart Online Grocery Shopping Dataset 2017”, Accessed from https://www.instacart.com/datasets/grocery-shopping-2017 on Retrieved 08-2018

please follow papers, dataset is available as part of Kaggle competition at https://www.kaggle.com/c/instacart-market-basket-analysis/data

2. Prepare tfrecords train and test data

python prepare_data.py

There problem with this data, it is hard to make is reasonably even. Starting date of customer shopping sequence is random, but aligned with day of week. Next dates are calculated iusing days between shopping. Random date is picked from an dates interval. Once all these dates are calculated, volume of all shopping reaches some maximum and then does down. This is how training data looks like. I selected grocery only items from products list. 

3. Training

When training see critic and actor and reward convergence. Also, at the end, replenishment average should be close to the sales 0.1 == 0.1

python training.py --batch_size=1024 --waste=0.05 --action=TRAIN --train_episodes=20000 --output_dir checkpoints

I consider waste should be 10% per day so 0.025 per timeinterval (just tried 0.05 here)

4. Evaluation

I did't evaluate. But see this output: there shouldn't be stockouts or overstocks and inventory should be as low ax possible.

![output sample](samples/curves/data_prep_cell_12_output_0.png "Sample inventory and replenishment dynamics")

i5. Prediction

time python training.py --action=PREDICT --output_dir checkpoints

This will produce output.csv with metrics for each timestep. if taking one product and following it over time, metrics will look like this

![output sample](samples/curves/data_prep_cell_12_output_0.png "Sample inventory and replenishment dynamics")
