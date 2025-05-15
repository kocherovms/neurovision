# Cortical columns hosted within separate process
import os, math
from collections import defaultdict, namedtuple
from dataclasses import dataclass
import typing
import itertools
import sqlite3
import datetime
import multiprocessing as mp
import queue

import numpy as np
import cupy as cp
import pandas as pd

from utils import *
from hdc import *

###
class Engram(object):
    def __init__(self):
        self.hdv_bundle_uncapped_ = None
        self.hdv_bundle = None
        self.image_value = None
        self.image_ids = set()
        self.is_sealed = False
        self.updates_counter = 0

    @property
    def hdv_bundle_uncapped(self):
        return self.hdv_bundle_uncapped_

    @hdv_bundle_uncapped.setter
    def hdv_bundle_uncapped(self, h):
        self.hdv_bundle_uncapped_ = h
        self.hdv_bundle = xp.sign(h)
        self.updates_counter += 1

class CorticalColumn(object):
    def __init__(self):
        # engram = collection of related HDV aka prototype or collective image
        self.engrams = {} # key - index in engram_norms, value - Engram instance
        self.engram_norms = HdvArray(hdc.N, xp) # hot part of computation - normalized versions of engrams for fast cos sim calculation
        self.images_seen = 0

###

@dataclass
class Task:
    task_id: int
    op: str
    params: dict = None

@dataclass
class TaskResult:
    task_id: int
    payload: typing.Any = None

LOG = Logging()
HOST_ID = None
RNG = None
COLUMNS = {}
hdc = None
xp = None
xp_array_from_gpu = None
xp_array_to_gpu = None
train_db_con = None
test_db_con = None
df_train_images = None
df_test_images = None

def live(config_var, host_id, column_ids, inp_queue, out_queue):
    global hdc
    global xp, xp_array_from_gpu, xp_array_to_gpu
    global train_db_con, test_db_con, df_train_images, df_test_images
    global RNG
    
    HOST_ID = host_id
    LOG.push_prefix('HOST_ID', host_id)
    LOG(f'Starting host #{HOST_ID} for cortical columns [{', '.join(map(str, column_ids))}]')
    
    config = Config(config_var)
    RNG = np.random.default_rng()
    
    if cp.cuda.is_available():
        xp = cp.get_array_module(cp.empty(1))
        xp_array_from_gpu = lambda a: a.get() if isinstance(a, cp.ndarray) else a
        xp_array_to_gpu = lambda a: cp.asarray(a) if isinstance(a, np.ndarray) else a
    else:
        xp = cp.get_array_module(np.empty(1))
        xp_array_from_gpu = lambda a: a
        xp_array_to_gpu = lambda a: a
        
    LOG(f'xp = {xp.__name__}')

    hdc = Hdc(10_000, xp)

    get_full_db_file_name = lambda db_file_name: os.path.join(config.dataset_path, config.db_file_name_prefix + db_file_name)
    train_db_con = sqlite3.connect(get_full_db_file_name(config.train_db_file_name))
    test_db_con = sqlite3.connect(get_full_db_file_name(config.test_db_file_name))
    df_train_images = pd.read_sql_query('SELECT * FROM images', con=train_db_con, index_col='image_id')
    df_test_images = pd.read_sql_query('SELECT * FROM images', con=test_db_con, index_col='image_id')

    LOG('Connected to databases')

    COLUMNS.update(map(lambda i: (i, CorticalColumn()), column_ids))
    LOG(f'Columns [{', '.join(map(str, column_ids))}] created')
    LOG(f'Host is ready')
    
    task_wait_timeout = 60
    is_running = True

    while is_running:
        try:
            # task is expected to be an instanace of Task class
            task = inp_queue.get(block=True, timeout=task_wait_timeout)
        except queue.Empty:
            LOG(f'Didn\'t get any tasks within {task_wait_timeout} seconds, waiting again')
            continue

        LOG.push_prefix('TASK_ID', task.task_id)
        LOG(f'Got task #{task.task_id} {task.op}')
        task_result = TaskResult(task_id=task.task_id)
        
        match task.op:
            case 'HEALTHCHECK':
                pass
            case 'TRAIN':
                train(task.params['train_run_id'], 
                      task.params['image_ids'], 
                      task.params['consolidation_threshold'], 
                      task.params['attempts_to_get_no_mistakes'])
            case 'INFER':
                column_votes_vector, column_images_seen = infer(task.params['dataset_name'], task.params['image_id'])
                task_result.payload = {'column_votes_vector': column_votes_vector, 'column_images_seen': column_images_seen}
            case 'DUMP':
                pass
            case 'TERMINATE':
                is_running = False
            case _:
                LOG(f'Unknown task op: {task.op}, ignoring')

        out_queue.put(task_result)
        LOG('Task complete')
        LOG.pop_prefix('TASK_ID')

    LOG(f'Host is going down')

def train(train_run_id, image_ids, consolidation_threshold, attempts_to_get_no_mistakes):
    LOG.push_prefix('TRRID', train_run_id)
    LOG(f'''Train params: len(image_ids) = {len(image_ids)}, 
    consolidation_threshold={consolidation_threshold}, attempts_to_get_no_mistakes={attempts_to_get_no_mistakes}''')
    
    for column_id, column in COLUMNS.items():
        LOG.push_prefix('COL', column_id)
        column_image_ids = image_ids.copy()
        
        for attempt_to_get_no_mistakes in range(attempts_to_get_no_mistakes):
            # 1) EVOLVE MEMORIES
            for image_no, image_id in enumerate(column_image_ids):
                # 1.1) MINE ENGRAMS
                LOG.push_prefix('IMGNO', image_no)
                LOG.push_prefix('IMGID', image_id)
                
                LOG(f'Engrams count={column.engram_norms.len}')
                column.images_seen += 1
                
                image_value = df_train_images.loc[image_id]['value']
                df_image_encodings = pd.read_sql('SELECT hdv FROM image_encodings WHERE image_id=:image_id AND column_id=:column_id', 
                                                 params={'image_id': int(image_id), 'column_id': column_id}, con=train_db_con)
                assert len(df_image_encodings) > 0
                image_encoding_hdvs = list(map(lambda h: np.frombuffer(h, dtype='b'), df_image_encodings['hdv']))
                image_encoding_hdvs_norm = hdc.normalize(image_encoding_hdvs)
                image_encoding_hdvs_norm = xp_array_to_gpu(image_encoding_hdvs_norm)
        
                cos_sim_matrix = column.engram_norms.array_active @ image_encoding_hdvs_norm.T
                cos_sim_matrix[cos_sim_matrix < Hdc.COS_SIM_THRESHOLD] = 0
                cos_sim_vector = xp_array_from_gpu(xp.sum(cos_sim_matrix, axis=1)) # how each mem recall (sum cos sim) is close to current image
                
                assert cos_sim_vector.shape == (column.engram_norms.array_active.shape[0],)
                engram_ids_by_match_score = np.argsort(-cos_sim_vector) # sorted desc
                match_found = False
                match_pos = False
                match_is_updated = False
        
                for pos, engram_id in enumerate(engram_ids_by_match_score):
                    cos_sim_value = cos_sim_vector[engram_id]
                    LOG(f'Checking engram #{engram_id}, pos={pos}, sim={cos_sim_value:.2f}')
        
                    if cos_sim_value <= 0:
                        break

                    engram = column.engrams[engram_id]
        
                    if engram.image_value != image_value:
                        LOG(f'Match WRONG, engram value={engram.image_value} vs {image_value}')
                    else:
                        LOG(f'Match CORRECT, sealed={engram.is_sealed}')
        
                        if not engram.is_sealed:
                            engram.image_ids.add(image_id)
                            image_encoding_hdv_bundle = hdc.bundle(image_encoding_hdvs)
                            
                            engram.hdv_bundle_uncapped = xp.sum(xp.vstack([engram.hdv_bundle_uncapped, image_encoding_hdv_bundle]), axis=0)
                            column.engram_norms.array[engram_id] = hdc.normalize(engram.hdv_bundle)
                            
                            if engram.updates_counter > 200:
                                LOG(f'engram #{engram_id} (upd. counter={engram.updates_counter}) is sealed')
                                engram.is_sealed = True

                            match_is_updated = True
        
                        match_found = True
                        match_pos = pos
                        break
        
                LOG(f'Match found={match_found}, pos={match_pos}, is updated={match_is_updated}')
        
                if (match_found and match_pos == 0) or (match_found and match_is_updated):
                    pass
                else:
                    # Deploy new engram
                    engram_id = column.engram_norms.lease()
                    
                    engram = Engram()
                    engram.hdv_bundle_uncapped = xp.sum(xp.vstack(image_encoding_hdvs), axis=0)
                    engram.image_ids.add(image_id)
                    engram.image_value = image_value
                    
                    column.engram_norms.array[engram_id] = hdc.normalize(engram.hdv_bundle)
                    column.engrams[engram_id] = engram
                    LOG(f'New engram {engram_id}')

                # 1.2) CONSOLIDATE ENGRAMS
                if (image_no + 1) % consolidation_threshold == 0:
                    def dump_engram_lengths(when):
                        engram_lengths = list(map(lambda e: len(e.image_ids), column.engrams.values()))
                        engram_lengths = np.unique_counts(engram_lengths)
                        
                        for v, c in zip(reversed(engram_lengths.values), reversed(engram_lengths.counts)):
                            LOG(f'[CONSOLIDATION] {when:10} {v:5} {c:5}')
        
                    before_len = column.engram_norms.len
                    assert before_len == len(column.engrams)
                    
                    LOG(f'[CONSOLIDATION] BEFORE engrams count = {before_len}')
                    dump_engram_lengths('BEFORE')
                    
                    engram_ids_to_release = []
                    
                    for engram_id, engram in column.engrams.items():
                        #l = 0.1
                        l = 1
                        exp_distro_level = l * np.exp(-l * len(engram.image_ids))
                        rand_level = RNG.random()
                        do_release = rand_level < exp_distro_level
                        
                        if do_release:
                            engram_ids_to_release.append(engram_id)
                            LOG(f'[CONSOLIDATION] Dropping engram #{engram_id}, len={len(engram.image_ids)}: {rand_level:.5f} < {exp_distro_level:.5f}')
                
                    for engram_id_to_release in engram_ids_to_release:
                        del column.engrams[engram_id_to_release]
                        column.engram_norms.release(engram_id_to_release)
                
                    after_len = column.engram_norms.len
                    assert after_len == len(column.engrams)
                    
                    LOG(f'[CONSOLIDATION] AFTER engrams count = {after_len}')
                    dump_engram_lengths('AFTER')
            
            LOG.pop_prefix('IMGNO')
            LOG.pop_prefix('IMGID')
            
            # 2) DETECT MISTAKES FOR CONSEQUENT FIX
            mistake_image_ids = []
            
            # for image_no, image_id in enumerate(column_image_ids):
            for image_no, image_id in enumerate(image_ids):
                image_value = df_train_images.loc[image_id]['value']
                df_image_encodings = pd.read_sql('SELECT hdv FROM image_encodings WHERE image_id=:image_id AND column_id=:column_id', 
                                                 params={'image_id': int(image_id), 'column_id': column_id}, con=train_db_con)
                assert len(df_image_encodings) > 0
                image_encoding_hdvs = list(map(lambda h: np.frombuffer(h, dtype='b'), df_image_encodings['hdv']))
                image_encoding_hdvs_norm = hdc.normalize(image_encoding_hdvs)
                image_encoding_hdvs_norm = xp_array_to_gpu(image_encoding_hdvs_norm)
            
                cos_sim_matrix = column.engram_norms.array_active @ image_encoding_hdvs_norm.T
                cos_sim_matrix[cos_sim_matrix < Hdc.COS_SIM_THRESHOLD] = 0
                cos_sim_vector = xp_array_from_gpu(xp.sum(cos_sim_matrix, axis=1)) # how each mem recall (sum cos sim) is close to current image
                
                assert cos_sim_vector.shape == (column.engram_norms.array_active.shape[0],)
                engram_ids_by_match_score = np.argsort(-cos_sim_vector) # sorted desc
                is_mistake = True
            
                if engram_ids_by_match_score.shape[0] > 0:
                    engram_id = engram_ids_by_match_score[0]
                    cos_sim_value = cos_sim_vector[engram_id]
            
                    if cos_sim_value > 0:
                        engram_image_value = column.engrams[engram_id].image_value
                        is_mistake = engram_image_value != image_value

                if is_mistake:
                    # no infer or incorrect infer
                    mistake_image_ids.append(image_id)

            LOG(f'Mistaken image ids = {len(mistake_image_ids)}')
            column_image_ids = mistake_image_ids

        if column_image_ids:
            LOG(f'{len(column_image_ids)} image ids remained after run')
            
    LOG.pop_prefix('TRRID')

def infer(dataset_name, image_id):
    datasets = {'train': (df_train_images, train_db_con), 'test': (df_test_images, test_db_con)}
    dataset = datasets[dataset_name]
    column_votes_vector = np.zeros(10)
    column_images_seen = {}
    
    for column_id, column in COLUMNS.items():
        max_cos_sim_index = -1 # aka engram id
        max_similar_engram_image_value = ''
        max_cos_sim = 0
    
        df_image_encodings = pd.read_sql('SELECT hdv FROM image_encodings WHERE image_id=:image_id AND column_id=:column_id', 
                                         params={'image_id': int(image_id), 'column_id': column_id}, con=dataset[1])
        assert len(df_image_encodings) > 0
        image_encoding_hdvs = list(map(lambda h: np.frombuffer(h, dtype='b'), df_image_encodings['hdv']))
        image_encoding_hdvs_norm = hdc.normalize(image_encoding_hdvs)
        image_encoding_hdvs_norm = xp_array_to_gpu(image_encoding_hdvs_norm)
    
        cos_sim_matrix = column.engram_norms.array_active @ image_encoding_hdvs_norm.T
        cos_sim_matrix[cos_sim_matrix < Hdc.COS_SIM_THRESHOLD] = 0
        cos_sim_vector = xp_array_from_gpu(xp.sum(cos_sim_matrix, axis=1)) # how each mem recall (sum cos sim) is close to current image
        
        assert cos_sim_vector.shape == (column.engram_norms.array_active.shape[0],)
        engram_ids_by_match_score = np.argsort(-cos_sim_vector) # sorted desc
    
        if engram_ids_by_match_score.shape[0] > 0:
            engram_id = engram_ids_by_match_score[0]
            cos_sim_value = cos_sim_vector[engram_id]
    
            if cos_sim_value > 0:
                max_cos_sim_index = engram_id
                max_similar_engram_image_value = column.engrams[engram_id].image_value
                max_cos_sim = cos_sim_value
                column_votes_vector[int(max_similar_engram_image_value)] += max_cos_sim

        column_images_seen[column_id] = column.images_seen

    return column_votes_vector, column_images_seen
