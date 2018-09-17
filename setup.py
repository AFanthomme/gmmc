# Please run this to create necessary folders that were git excluded

import os

try:
    f = open('out/test_out_exists.txt', 'w+')
except FileNotFoundError:
    os.makedirs('out')

try:
    f = open('out/raw/test_out_exists.txt', 'w+')
except FileNotFoundError:
    os.makedirs('out/raw')

try:
    f = open('out/processed/test_out_exists.txt', 'w+')
except FileNotFoundError:
    os.makedirs('out/processed')
