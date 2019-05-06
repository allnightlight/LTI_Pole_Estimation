
import os
import pickle
import itertools
import sqlite3
from modules import *
from datetime import datetime, timedelta

conn = sqlite3.connect('db.sqlite')
cur = conn.cursor()

# initialize data generator
Nhidden = 2**4
data_generator = DataGenerator(Nhidden = Nhidden, Ntrain = 2**10)
Ny, Nu = data_generator.Ny, data_generator.Nu

cur.execute('''Select count(id) from Data ''')
cnt = cur.fetchone()[0]

lti_file_path_ = './tmp/data_%03d.pk' % cnt
with open(lti_file_path_, 'wb') as fp:
    pickle.dump(data_generator, fp)
lti_file_path = [lti_file_path_,]

# run training
modelclass = ['model001', 'model002', 'model003', 'model004', ]
Nbatch = [2**5, ]
Nepoch = [2**6, 2**8, 2**10, ]
Nhrz = [2**0, 2**2, 2**4, ]
Nhidden = [Nhidden, 2*Nhidden]

t_bgn = datetime.now()
for modelclass_, lti_file_path_, Nbatch_, Nepoch_, Nhrz_, Nhidden_ in itertools.cycle(itertools.product(
    modelclass, lti_file_path, Nbatch, Nepoch, Nhrz, Nhidden)):

    elapsed_time = datetime.now() - t_bgn
    if elapsed_time  > timedelta(minutes = 30):
        break

    try:
        with open(lti_file_path_, "rb") as fp:
            data_generator_ = pickle.load(fp)

        mdl, _ = run_training(lambda Nhidden: eval(modelclass_)(Ny, Nu, Nhidden), 
            data_generator_, Nhidden_, Nbatch_, Nepoch_, Nhrz_)
        print('\n')
    except:
        continue

# 
    cur.execute('''Insert or Ignore into ModelClass (name) values (?)''', (modelclass_,))
    cur.execute('''Insert or Ignore into Data (lti_file_path) values (?)''', (lti_file_path_,))
    cur.execute('''Insert or Ignore into Training (Nbatch, Nepoch, Nhrz) values (?,?,?)''', 
        (Nbatch_, Nepoch_, Nhrz_))
    cur.execute('''Select Count(id) from Result''')
    cnt = cur.fetchone()[0]
    
    model_file_path = "./tmp/%s_%04d.pt" % (modelclass_, cnt)
    torch.save(mdl.state_dict(), model_file_path)

    cur.execute('''Select id from ModelClass where name = ?''', (modelclass_,))
    modelclass_id = cur.fetchone()[0]

    cur.execute('''Select id from Training where  Nbatch = ? and Nepoch = ? and Nhrz = ?''', 
        (Nbatch_, Nepoch_, Nhrz_,))
    training_id = cur.fetchone()[0]

    cur.execute('''Select id from Data where lti_file_path = ?''', (lti_file_path_,))
    data_id = cur.fetchone()[0]
    
    cur.execute('''Insert into Result (training_id, data_id, model_file_path, Nhidden, modelclass_id) 
        values (?, ?, ?, ?, ?) ''', 
        (training_id, data_id, model_file_path, Nhidden_, modelclass_id))

    conn.commit()
conn.close()
