import torch
import utils
import loader
import eeppr
import logger
import main
print(
    'Testing if all libraries are installed and can be imported successfully... ', end='')
torch._C._cuda_init()
print('All import were successful!')
print('OK')
