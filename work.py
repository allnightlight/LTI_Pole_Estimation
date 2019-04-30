
import matplotlib.pylab as plt
import pickle
import sqlite3
from modules import *
conn = sqlite3.connect('db.sqlite')
cur = conn.cursor()

cur.execute('''Select Result.model_file_path, Result.Nhidden, Data.lti_file_path,
    Training.Nepoch, Training.Nhrz
    from Result join Data join Training
    where Result.data_id = Data.id and Result.training_id = Training.id
    and Data.lti_file_path = (?)
    ''', ('./tmp/data_005.pk',))

tbl = []
for (model_file_path, Nhidden, lti_file_path, Nepoch, Nhrz) in cur.fetchall():
    with open(lti_file_path, "rb") as fp:
        data_generator = pickle.load(fp)
    Ny = data_generator.Ny
    Nu = data_generator.Nu
    mdl = model001(Nhidden, Ny, Nu)
    mdl.load_state_dict(torch.load(model_file_path))

    _weight = mdl.xu2x.weight
    weight = _weight.data.numpy()
    A_hat = weight[:, :Nhidden]
    A = data_generator.A

    wdist = check_spectrum(A, A_hat)
    tbl.append((Nepoch, Nhrz, wdist))

tbl = np.array(tbl)
plt.plot(tbl[:,0], tbl[:,2], 'x')
plt.yscale('log')
plt.show()



