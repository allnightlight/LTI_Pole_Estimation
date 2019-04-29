
import os
import pickle
import itertools
import sqlite3
from work001modules import *
from datetime import datetime, timedelta

conn = sqlite3.connect('work001db.sqlite')
cur = conn.cursor()

# initialize data generator
Ny = 1
Nu = 2
data_generator = DataGenerator()
lti_file_path = './work001tmp/data001.pk'
with open(lti_file_path,'wb') as fp:
    pickle.dump(data_generator, fp)

# run training
mdl_constructor = [lambda Nhidden: model001(Ny,Nu,Nhidden), ]
lti_file_path = ['./work001tmp/data001.pk',]
Nbatch = [2**5,]
Nepoch = [2**2,]
Nhrz = [2**0, 2**1, 2**2, 2**3,]
Nhidden = [2**0,]

t_bgn = datetime.now()
for mdl_constructor_, lti_file_path_, Nbatch_, Nepoch_, Nhrz_, Nhidden_ in itertools.cycle(itertools.product(
    mdl_constructor, lti_file_path, Nbatch, Nepoch, Nhrz, Nhidden)):

    elapsed_time = datetime.now() - t_bgn
    if elapsed_time  > timedelta(seconds = 10):
        break

    try:
        with open(lti_file_path_, "rb") as fp:
            data_generator_ = pickle.load(fp)
        mdl, _ = run_training(mdl_constructor_, data_generator_, Nhidden_, Nbatch_, Nepoch_, Nhrz_)
    except:
        continue

# 
    cur.execute('''Insert or Ignore into Data (lti_file_path) values (?)''', (lti_file_path_,))
    cur.execute('''Insert or Ignore into Training (Nbatch, Nepoch, Nhrz) values (?,?,?)''', 
        (Nbatch_, Nepoch_, Nhrz_))
    cur.execute('''Select Count(id) from Result''')
    cnt = cur.fetchone()[0]
    
    model_file_path = "./work001tmp/model_%04d.pt" % cnt
    torch.save(mdl.state_dict(), model_file_path)

    cur.execute('''Select id from Training where  Nbatch = ? and Nepoch = ? and Nhrz = ?''', 
        (Nbatch_, Nepoch_, Nhrz_,))
    training_id = cur.fetchone()[0]

    cur.execute('''Select id from Data where lti_file_path = ?''', (lti_file_path_,))
    data_id = cur.fetchone()[0]
    
    cur.execute('''Insert into Result (training_id, data_id, model_file_path, Nhidden) values (?, ?, ?, ?) ''', 
        (training_id, data_id, model_file_path, Nhidden_))
