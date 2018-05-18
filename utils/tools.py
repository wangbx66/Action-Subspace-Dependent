from os import path

#import ipdb
#debug = ipdb.set_trace
debug = None

import socket
host = socket.gethostname()
if (len(host) == 10) and (host[:7] == 'compute'):
    mjkey_mutex = True
else:
    mjkey_mutex = False

def wait_mutex():
    import time
    while True:
        try:
            sign = open(os.path.expanduser('~/.mujoco/mutex')).readline()
            if not sign == 'available':
                print('Mutex locked')
                time.sleep(1)
            else:
                return
        except FileNotFoundError:
            print('Mutex not found')
            time.sleep(1)

def hold_mutex():
    with open(os.path.expanduser('~/.mujoco/mutex'), 'w') as fw:
        fw.write('occupied')

def release_mutex():
    with open(os.path.expanduser('~/.mujoco/mutex'), 'w') as fw:
        fw.write('available')

def get_key():
    assert(host[:7] == 'compute') and (len(host) == 10)
    computer = host[-3:]
    mjkey = os.path.expanduser('~/.mujoco/mjkey{}.txt'.format(computer))
    from shutil import copyfile
    copyfile(mjkey, os.path.expanduser('~/.mujoco/mjkey.txt'))

def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))

