from __future__ import print_function
#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import random as rn

### We modified Pahikkala et al. (2014) source code for cross-val process ###

import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  #指定要使用的GPU序号

np.random.seed(1)
rn.seed(1)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth=True   	  #不全部占满显存, 动态增长
from tensorflow import keras
from tensorflow.keras import backend as K
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from datahelper import *
#import logging
from itertools import product
from arguments import argparser, logging

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv2D, GRU, CuDNNGRU, CuDNNLSTM
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, Flatten
#from tensorflow.summary import merge
from tensorflow.keras.models import Model
#from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import optimizers, layers

import sys, pickle
import math, json, time
import decimal
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing
from emetrics import get_aupr, get_cindex, get_rm2

def build_combined_onehot(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    XDinput = Input(shape=(FLAGS.max_smi_len, FLAGS.charsmiset_size))
    XTinput = Input(shape=(FLAGS.max_seq_len, FLAGS.charseqset_size))

    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, padding='valid', strides=1, dilation_rate=2)(XDinput)
    encode_smiles = LeakyReLU(alpha=0.1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1, padding='valid', strides=1, dilation_rate=2)(encode_smiles)
    encode_smiles = LeakyReLU(alpha=0.1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1, padding='valid', strides=1, dilation_rate=2)(encode_smiles)
    encode_smiles = LeakyReLU(alpha=0.1, name='CNNfeature_smi')(encode_smiles)
    encode_smiles = GlobalMaxPooling1D(name='MaxPoolfeature_smi')(encode_smiles)
    encode_smiles_gru = Bidirectional(CuDNNLSTM(units=NUM_FILTERS*3), merge_mode='ave', name='LSTMfeature_smi')(XDinput)
    encode_smiles = keras.layers.concatenate([encode_smiles, encode_smiles_gru])
     #pool_size=pool_length[i]
    
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, padding='valid', strides=1, dilation_rate=2)(XTinput)
    encode_protein = LeakyReLU(alpha=0.1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2, padding='valid', strides=1, dilation_rate=2)(encode_protein)
    encode_protein = LeakyReLU(alpha=0.1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2, padding='valid', strides=1, dilation_rate=2)(encode_protein)
    encode_protein = LeakyReLU(alpha=0.1, name='CNNfeature_pro')(encode_protein)
    encode_protein = GlobalMaxPooling1D(name='MaxPoolfeature_pro')(encode_protein)
    encode_protein_gru = Bidirectional(CuDNNLSTM(units=NUM_FILTERS*3), merge_mode='ave', name='LSTMfeature_pro')(XTinput)
    encode_protein = keras.layers.concatenate([encode_protein, encode_protein_gru])
    
    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], name='ConcatFeature')
    #encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected 
    
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu', name='DenseFeature')(FC2)
    predictions = Dense(1, kernel_initializer='normal')(FC2) 

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    
    print(interactionModel.summary())
    return interactionModel


def build_combined_categorical(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
   
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32') ### Buralar flagdan gelmeliii
    XDstru_input = Input(shape=(FLAGS.max_smi_stru_len,), dtype='int32')
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')
    XTstru_input = Input(shape=(FLAGS.max_seq_len,), dtype='int32')


    ### SMI_EMB_DINMS  FLAGS GELMELII 
    encode_smiles_embed = Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)(XDinput) 
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, padding='valid', strides=1, dilation_rate=2)(encode_smiles_embed)
    encode_smiles = LeakyReLU(alpha=0.1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1, padding='valid', strides=1, dilation_rate=2)(encode_smiles)
    encode_smiles = LeakyReLU(alpha=0.1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1, padding='valid', strides=1, dilation_rate=2)(encode_smiles)
    encode_smiles = LeakyReLU(alpha=0.1, name='CNNfeature_smi')(encode_smiles)
    encode_smiles = GlobalMaxPooling1D(name='MaxPoolfeature_smi')(encode_smiles)
    encode_smiles_gru = Bidirectional(CuDNNLSTM(units=NUM_FILTERS*3), merge_mode='ave', name='LSTMfeature_smi')(encode_smiles_embed)
    encode_smiles = keras.layers.concatenate([encode_smiles, encode_smiles_gru])

    encode_smiles_stru_embed = Embedding(input_dim=512, output_dim=128, input_length=FLAGS.max_smi_stru_len)(XDstru_input) 
    encode_smiles_stru = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, padding='valid', strides=1, dilation_rate=2)(encode_smiles_stru_embed)
    encode_smiles_stru = LeakyReLU(alpha=0.1)(encode_smiles_stru)
    encode_smiles_stru = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1, padding='valid', strides=1, dilation_rate=2)(encode_smiles_stru)
    encode_smiles_stru = LeakyReLU(alpha=0.1)(encode_smiles_stru)
    encode_smiles_stru = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1, padding='valid', strides=1, dilation_rate=2)(encode_smiles_stru)
    encode_smiles_stru = LeakyReLU(alpha=0.1, name='CNNfeature_smi_stru')(encode_smiles_stru)
    encode_smiles_stru = GlobalMaxPooling1D(name='MaxPoolfeature_smi_stru')(encode_smiles_stru)

    encode_protein_embed = Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128, input_length=FLAGS.max_seq_len)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, padding='valid', strides=1, dilation_rate=2)(encode_protein_embed)
    encode_protein = LeakyReLU(alpha=0.1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2, padding='valid', strides=1, dilation_rate=2)(encode_protein)
    encode_protein = LeakyReLU(alpha=0.1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2, padding='valid', strides=1, dilation_rate=2)(encode_protein)
    encode_protein = LeakyReLU(alpha=0.1, name='CNNfeature_pro')(encode_protein)
    encode_protein = GlobalMaxPooling1D(name='MaxPoolfeature_pro')(encode_protein)
    encode_protein_gru = Bidirectional(CuDNNLSTM(units=NUM_FILTERS*3), merge_mode='ave', name='LSTMfeature_pro')(encode_protein_embed)
    encode_protein = keras.layers.concatenate([encode_protein, encode_protein_gru])

    encode_protein_stru_embed = Embedding(input_dim=FLAGS.charseqstruset_size+1, output_dim=128, input_length=FLAGS.max_seq_len)(XTstru_input)
    encode_protein_stru = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, padding='valid', strides=1, dilation_rate=2)(encode_protein_stru_embed)
    encode_protein_stru = LeakyReLU(alpha=0.1)(encode_protein_stru)
    encode_protein_stru = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2, padding='valid', strides=1, dilation_rate=2)(encode_protein_stru)
    encode_protein_stru = LeakyReLU(alpha=0.1)(encode_protein_stru)
    encode_protein_stru = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2, padding='valid', strides=1, dilation_rate=2)(encode_protein_stru)
    encode_protein_stru = LeakyReLU(alpha=0.1, name='CNNfeature_pro_stru')(encode_protein_stru)
    encode_protein_stru = GlobalMaxPooling1D(name='MaxPoolfeature_pro_stru')(encode_protein_stru)

    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein, encode_protein_stru, encode_smiles_stru], name='ConcatFeature')
    #encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected 
    
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu', name='DenseFeature')(FC2)
    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC2) #OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

    interactionModel = Model(inputs=[XDinput, XDstru_input, XTinput, XTstru_input], outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    print(interactionModel.summary())

    return interactionModel

def nfold_1_2_3_setting_sample(XD, XD_STRU, XT, XT_STRU, Y, label_row_inds, label_col_inds, measure, runmethod,  FLAGS, dataset):
    test_set, outer_train_sets = dataset.read_sets(FLAGS) 
    foldinds = len(outer_train_sets)
    test_sets = []
    val_sets = []
    train_sets = []
    bestpar_list = []
    bestprf_list = []

    test_sets.append(test_set)
    for ep, val_foldind in enumerate(range(foldinds)):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds)
        print("val set", str(len(val_fold)))
        print("train set", str(len(otherfoldsinds)))

        bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv(XD, XD_STRU, XT, XT_STRU, Y, label_row_inds, label_col_inds, 
                                                                                                measure, runmethod, FLAGS, train_sets, val_sets, ep+1)
        
    bestpar_list.append(bestparamind)
    bestprf_list.append(bestperf)
    bestind = np.argmax(bestprf_list)
    test_prf, test_loss = best_model_test(XD, XD_STRU, XT, XT_STRU, Y, label_row_inds, label_col_inds, FLAGS, measure, test_sets, bestind, bestpar_list)
    return test_prf, test_loss

def general_nfold_cv(XD, XD_STRU, XT, XT_STRU, Y, label_row_inds, label_col_inds, prfmeasure, runmethod, FLAGS, labeled_sets, val_sets, ep):
    NumWinParam = FLAGS.num_windows   #paramset1
    SmiWinLen = FLAGS.smi_window_lengths   #paramset2
    SeqWinLen = FLAGS.seq_window_lengths   #paramset3
    epoch = FLAGS.num_epoch 
    batchsz = FLAGS.batch_size

    w = len(val_sets)
    h = len(NumWinParam) * len(SmiWinLen) * len(SeqWinLen)
    all_predictions = [[0 for x in range(w)] for y in range(h)] 
    all_losses = [[0 for x in range(w)] for y in range(h)]

    for foldind in range(len(val_sets)):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]

        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]
        train_drugs, train_drugs_stru, train_prots, train_prots_stru, train_Y = prepare_interaction_pairs(XD, XD_STRU, XT, XT_STRU, Y, trrows, trcols)

        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]
        val_drugs, val_drugs_stru, val_prots, val_prots_stru, val_Y = prepare_interaction_pairs(XD, XD_STRU, XT, XT_STRU, Y, terows, tecols)

        pointer = 0
        for numwinvalue in NumWinParam:
            for smiwinlenvalue in SmiWinLen:
                for seqwinlenvalue in SeqWinLen:
                    model = runmethod(FLAGS, numwinvalue, smiwinlenvalue, seqwinlenvalue)
                    my_callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15),
                                    TensorBoard(log_dir=FLAGS.log_dir)]

                    model.fit(([np.array(train_drugs),np.array(train_drugs_stru),np.array(train_prots),np.array(train_prots_stru)]), np.array(train_Y), batch_size=batchsz, epochs=epoch, 
                    validation_data=(([np.array(val_drugs), np.array(val_drugs_stru), np.array(val_prots),np.array(val_prots_stru)]), np.array(val_Y)),  shuffle=False, callbacks=my_callbacks)
                    
                    model.save(f'{pointer}_{ep}_model.h5')
                    pred = model.predict([np.array(val_drugs), np.array(val_drugs_stru), np.array(val_prots),np.array(val_prots_stru)])
                    loss, prf2 = model.evaluate(([np.array(val_drugs), np.array(val_drugs_stru), np.array(val_prots),np.array(val_prots_stru)]), np.array(val_Y), verbose=0)
                    prf = prfmeasure(val_Y, pred)
                    logging("ValidSets: P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI-i = %f, CI-ii = %f, MSE = %f" % 
                    (numwinvalue, smiwinlenvalue, seqwinlenvalue, foldind, prf, prf2, loss), FLAGS)

                    all_predictions[pointer][foldind] = prf #TODO FOR EACH VAL SET allpredictions[pointer][foldind]
                    all_losses[pointer][foldind]= loss
                    pointer +=1

    bestperf = -float('Inf')
    bestpointer = None
    best_param_list = []
    pointer = 0
    for numwinvalue in NumWinParam:
            for smiwinlenvalue in SmiWinLen:
                for seqwinlenvalue in SeqWinLen:
                    avgperf = 0.
                    for foldind in range(len(val_sets)):
                        foldperf = all_predictions[pointer][foldind]
                        avgperf += foldperf
                    avgperf /= len(val_sets)
                    if avgperf > bestperf:
                        bestperf = avgperf
                        bestpointer = pointer
                        best_param_list = [numwinvalue, smiwinlenvalue, seqwinlenvalue]

                    pointer +=1
    return  bestpointer, best_param_list, bestperf, all_predictions, all_losses

def best_model_test(XD, XD_STRU, XT, XT_STRU, Y, label_row_inds, label_col_inds, FLAGS, measure, test_sets, bestpointer, bestpar_list):
    model = keras.models.load_model(f'{bestpar_list[bestpointer]}_{bestpointer+1}_model.h5', custom_objects={'cindex_score': cindex_score})
    logging("LoadModel: pointer = %f, ep = %f" % (bestpar_list[bestpointer], bestpointer+1), FLAGS)

    terows = label_row_inds[test_sets]
    tecols = label_col_inds[test_sets]
    test_drugs, test_drugs_stru, test_prots, test_prots_stru, test_Y = prepare_interaction_pairs(XD, XD_STRU, XT, XT_STRU, Y, terows, tecols)
    pred = model.predict([np.array(test_drugs), np.array(test_drugs_stru), np.array(test_prots), np.array(test_prots_stru)])
    loss, prf2 = model.evaluate(([np.array(test_drugs), np.array(test_drugs_stru), np.array(test_prots), np.array(test_prots_stru)]), np.array(test_Y), verbose=0)
    prf = measure(test_Y, pred)
    logging("TestSets: CI-i = %f, CI-ii = %f, MSE = %f" % (prf, prf2, loss), FLAGS)

    return prf, loss

def cindex_score(y_true, y_pred):

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select

def prepare_interaction_pairs(XD, XD_STRU, XT, XT_STRU, Y, rows, cols):
    drugs = []
    drugs_stru = []
    targets = []
    targets_stru = []
    affinity=[] 
        
    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        drug_stru = XD_STRU[rows[pair_ind]]
        drugs_stru.append(drug_stru)

        target=XT[cols[pair_ind]]
        targets.append(target)

        target_stru = XT_STRU[cols[pair_ind]]
        targets_stru.append(target_stru)

        affinity.append(Y[rows[pair_ind],cols[pair_ind]])

    drug_data = np.stack(drugs)
    drug_stru_data = np.stack(drugs_stru)
    target_data = np.stack(targets)
    target_stru_data = np.stack(targets_stru)

    return drug_data, drug_stru_data, target_data, target_stru_data, affinity

def experiment(FLAGS, perfmeasure, deepmethod, foldcount=6): #5-fold cross validation + test

    #Input
    #XD: [drugs, features] sized array (features may also be similarities with other drugs
    #XT: [targets, features] sized array (features may also be similarities with other targets
    #Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
    #perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    #higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    #foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation


    dataset = DataSet( fpath = FLAGS.dataset_path, ### BUNU ARGS DA GUNCELLE
                      setting_no = FLAGS.problem_type, ##BUNU ARGS A EKLE
                      seqlen = FLAGS.max_seq_len,
                      smilen = FLAGS.max_smi_len,
                      smistrulen = FLAGS.max_smi_stru_len,
                      need_shuffle = False )
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size 
    FLAGS.charsmiset_size = dataset.charsmiset_size 
    FLAGS.charseqstruset_size = dataset.charseqstruset_size

    XD, XD_STRU, XT, XT_STRU, Y = dataset.parse_data(FLAGS)

    XD = np.asarray(XD)
    XD_STRU = np.asarray(XD_STRU)
    XT = np.asarray(XT)
    XT_STRU = np.asarray(XT_STRU)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    print(drugcount)
    targetcount = XT.shape[0]
    print(targetcount)

    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)  #basically finds the point address of affinity [x,y]

    S1_avgperf, S1_avgloss = nfold_1_2_3_setting_sample(XD, XD_STRU, XT, XT_STRU, Y, label_row_inds, label_col_inds,
                                                                     perfmeasure, deepmethod, FLAGS, dataset)

    logging("Setting " + str(FLAGS.problem_type), FLAGS)
    logging("avg_perf = %.5f,  avg_mse = %.5f " % 
            (S1_avgperf, S1_avgloss), FLAGS)




def run_regression( FLAGS ): 

    perfmeasure = get_cindex
    deepmethod = build_combined_categorical

    experiment(FLAGS, perfmeasure, deepmethod)


if __name__=="__main__":
    FLAGS = argparser()
    FLAGS.log_dir = str(FLAGS.log_dir) + '0915cuda0' + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    print(FLAGS.log_dir)
    logging(str(FLAGS), FLAGS)
    run_regression( FLAGS )



                    
                    


