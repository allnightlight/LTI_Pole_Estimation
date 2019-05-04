
import sqlite3
import os
import shutil

conn = sqlite3.connect('db.sqlite')
cur = conn.cursor()

cur.executescript('''
Drop Table If Exists ModelClass;
Drop Table If Exists Data;
Drop Table If Exists Result;
Drop Table If Exists Training;

Create Table ModelClass(
id Integer Primary Key Unique,
name Text Unique
);

Create Table Training (
id Integer Primary Key Unique,
Nbatch Integer,
Nepoch Integer,
Nhrz Integer,
Unique(Nbatch, Nepoch, Nhrz)
);

Create Table Data (
id Integer Primary Key Unique,
lti_file_path Text Unique,
data_file_path Text Unique
);

Create Table Result(
id Integer Primary Key,
training_id Integer,
data_id Integer,
model_file_path Text Unique,
Nhidden Integer, 
modelclass_id Integer
);

''')

tmp_folder_path = "./tmp"
if os.path.exists(tmp_folder_path):
    shutil.rmtree(tmp_folder_path)
os.mkdir(tmp_folder_path)


