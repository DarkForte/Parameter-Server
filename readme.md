#Simple parameter server on c++

##Prerequisites:
* cmake
* Protobuf 3.0+
* MPICH 2

##Building & run:
```shell
cmake .
make
python run.py [parameters]
```

##Parameters:
Required:
* `--server [server_num]` : number of servers
* `--worker [worker_num]` : number of workers
* `--path [path]` : path to the training/testing data
* `--train [train_file_name]` : the file name of training data
* `--test [test_file_name]` : the file name of test data
Optional:
* `--hosts [hosts]` : host list, see MPI Tutorials for details. If not specified, all processed will run on the local machine.
* `--learning_rate`: learning rate, default = 0.0001.
* `--iters`:number of iterations on each worker, default = 2000.
* `--batch_size`:default = 64.
* `--seed`:random seed.

##File format
Please use libsvm format for both training and test data, and put them in the same folder. Also, for current version, please create a `train_file_name.meta` and `test_file_name.meta` to specify the total feature number for each file.