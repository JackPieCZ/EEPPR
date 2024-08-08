print('Testing if all libraries are installed and can be imported successfully... ', end='')
import main
import logger
import ee3p3d
import loader
import utils
import torch
torch._C._cuda_init()
print('All import were successful!')
print('OK')
