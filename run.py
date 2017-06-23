import sys
import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--server", type = int, dest = "server_num", required=True)
parser.add_argument("--worker", type = int, dest = "worker_num", required=True)
parser.add_argument("--hosts", dest = "hosts", default="")
parser.add_argument("--path", dest = "path", required=True)
parser.add_argument("--train", dest="train_file_name", required=True)
parser.add_argument("--test", dest="test_file_name", required=True)
parser.add_argument("--learning_rate", type = float, default = 0.0001)
parser.add_argument("--iters", type = int, default=2000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

total_num = int(args.server_num) + int(args.worker_num) + 1

hosts_command = ""
if args.hosts != "":
  hosts_command = "--hosts " + args.hosts

command = "mpirun --np {} {} ./param_server {} {} {} {} {} {} {} {} {}"\
  .format(total_num, hosts_command, args.server_num, args.worker_num,
          args.path, args.train_file_name, args.test_file_name,
          args.learning_rate, args.iters, args.batch_size, args.seed)

print "Executing command: " + command

subprocess.call(command, shell=True)