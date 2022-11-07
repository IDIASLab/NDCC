# NDCC

We present the NDCC algorithm for noisy labeled data instances detection and counterfactual correction.


### Prerequisites
Python 3.6 or above and the following libraries
```
numpy
pickle
torch
torchvision
pandas
sklearn
tqdm
random
sys
```

## Files
```
  run_ndcc.py: main running file of the NDCC algorithm
  NDCC.py: main class of the NDCC; called by run_ndcc.py
  functionn.py: independed function needed in NDCC.py; called by NDCC.py
  acc_testing.py: testing the trained model
```

### How to use
```
Step 1. Make sure run_ndcc.py; NDCC.py; functionn.py under the same path

Step 2. Open the run_ndcc.py file

Step 3. Set the path of train/test file in the "Customer Input Parameters"
        Example: dataset_name = "CIFAR10"
        
Step 4. Set the a appropraite parameters in "Customer Input Parameters"
         Example: 
          input_data_size = 10000
          noise_rate = 0.4
          noise_type = "AS"
          cf_num_iteration = 25
          num_training_epoch = 10
          train_batch_size = 32
          test_batch_size = 32
          learning_rate = 0.1

Step 5. Running the run_ndcc.py file
```

