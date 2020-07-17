'''
Created on 2020/07/09

@author: ukai
'''
import os
import sqlite3

from store_field import StoreField


class Store(object):
    '''
    classdocs
    '''
    
    
    def __init__(self, dbPath):
        
        def create_db(dbPath):
            conn = sqlite3.connect(dbPath)
            cur = conn.cursor()
        
            cur.executescript("""
        
            Drop Table If Exists BuildParameter;
            Create Table BuildParameter(
                build_parameter_id Integer Primary Key,
                build_parameter_key Text Unique,                
                build_parameter_label Text,
                build_parameter_memento Text
            );
        
            Drop Table If Exists TrainLog;
            Create Table TrainLog(
                train_log_id Integer Primary Key,
                build_parameter_id Integer,
                agent_memento Text Unique,
                epoch Integer,
                Unique(build_parameter_id, epoch),
                FOREIGN KEY (build_parameter_id) REFERENCES BuildParameter (build_parameter_id) 
            );
        
            """)
       
            conn.commit() 
            conn.close()

        if not os.path.exists(dbPath):
            create_db(dbPath)
        
        self.dbPath = dbPath        
    
    def append(self, storeField):
        assert isinstance(storeField, StoreField)
        
        def myupdate(dbPath, build_parameter_key, build_parameter_label, build_parameter_memento, agent_memento, epoch):
            conn = sqlite3.connect(dbPath)
            cur = conn.cursor()
        
            cur.execute("""
            Insert Or Ignore Into BuildParameter (
                build_parameter_key
                , build_parameter_label
                , build_parameter_memento
                ) values (?, ?, ?)
            """, (build_parameter_key, build_parameter_label, build_parameter_memento,))
            cur.execute("""
            Select 
                build_parameter_id
                    From BuildParameter
                    Where build_parameter_key = ?
            """, (build_parameter_key,))
            build_parameter_id, = cur.fetchone()
        
            cur.execute("""
            Insert Or Ignore Into TrainLog (
                build_parameter_id
                , agent_memento
                , epoch
                ) values (?,?,?)
            """, (build_parameter_id, agent_memento, epoch,))
            
            conn.commit()
            conn.close()
                    
        myupdate(self.dbPath
                 , storeField.buildParameterKey
                 , storeField.buildParameterLabel
                 , storeField.buildParameterMemento
                 , storeField.agentMemento
                 , storeField.epoch)

    def restore(self, buildParameterLabel, epoch = None, buildParameterKey = None):
                
        def my_find_all(dbPath, buildParameterKey, buildParameterLabel, epoch):
            conn = sqlite3.connect(dbPath)
            cur = conn.cursor()
                    
            sql = """\
            Select
                t.agent_memento
                , t.epoch
                , b.build_parameter_memento
                , b.build_parameter_key
                , b.build_parameter_label
                From TrainLog t
                    Join BuildParameter b
                        On t.build_parameter_id = b.build_parameter_id
                Where b.build_parameter_label like ?"""
            if epoch is not None:
                sql += " And epoch = %d" % epoch
            if buildParameterKey is not None:
                sql += " And build_parameter_key = \"%s\"" % buildParameterKey
            sql += " Order by b.build_parameter_id, t.epoch"
                
            cur.execute(sql, (buildParameterLabel,))
                
            res = [*cur.fetchall()]
            conn.close()

            for agent_memento, epoch, buildParameterMemento, buildParameterKey, buildParameterLabel in res:
                yield (agent_memento, epoch, buildParameterMemento, buildParameterKey, buildParameterLabel)
        
        for (agent_memento, epoch, buildParameterMemento, buildParameterKey, buildParameterLabel) in my_find_all(self.dbPath, buildParameterKey, buildParameterLabel, epoch):
            yield StoreField(agent_memento, epoch, buildParameterMemento, buildParameterKey, buildParameterLabel)