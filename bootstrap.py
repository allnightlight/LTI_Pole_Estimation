
import os
import pickle
import itertools
import sqlite3
from modules import *
from datetime import datetime, timedelta

conn = sqlite3.connect('db.sqlite')
cur = conn.cursor()

# initialize data generator
data_generator = DataGenerator(Nhidden = 2**3, Ntrain = 2**10)
Ny, Nu = data_generator.Ny, data_generator.Nu

cur.execute('''Select count(id) from Data ''')
cnt = cur.fetchone()[0]

lti_file_path_ = './tmp/data_%03d.pk' % cnt
with open(lti_file_path_, 'wb') as fp:
    pickle.dump(data_generator, fp)
lti_file_path = [lti_file_path_,]

# run training
mdl_constructor = [lambda Nhidden: model001(Ny,Nu,Nhidden), ]
Nbatch = [2**5,]
Nepoch = [2**5, 2**7, 2**9]
Nhrz = [2**0, 2**3,]
Nhidden = [2**5,]

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
    
    model_file_path = "./tmp/model_%04d.pt" % cnt
    torch.save(mdl.state_dict(), model_file_path)

    cur.execute('''Select id from Training where  Nbatch = ? and Nepoch = ? and Nhrz = ?''', 
        (Nbatch_, Nepoch_, Nhrz_,))
    training_id = cur.fetchone()[0]

    cur.execute('''Select id from Data where lti_file_path = ?''', (lti_file_path_,))
    data_id = cur.fetchone()[0]
    
    cur.execute('''Insert into Result (training_id, data_id, model_file_path, Nhidden) values (?, ?, ?, ?) ''', 
        (training_id, data_id, model_file_path, Nhidden_))

    conn.commit()
conn.close()
