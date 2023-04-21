#!/bin/bash

# use zerotier network
rsync -r ./a1_gym_deploy/* unitree@172.29.85.13:~/a1_gym/a1_gym_deploy
