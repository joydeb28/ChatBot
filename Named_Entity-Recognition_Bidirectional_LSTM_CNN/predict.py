from prepro import createBatches,createMatrices,addCharInformatioin,padding
from nn import word2Idx, label2Idx, case2Idx, char2Idx, idx2Label
#from nn import entity_extract_model_BLSTM
import numpy as np

entity_extract_model_BLSTM = None

def getModel():
    global entity_extract_model_BLSTM
    if None==entity_extract_model_BLSTM:
        from nn import entity_extract_model_BLSTM

def predict(sentence,model):
    sen_list = [[[i,'O\n'] for i in sentence.split()]]
    #sen_list = [[['SOCCER', 'O\n'], ['-', 'O\n'], ['JAPAN', 'O\n'], ['GET', 'O\n'], ['LUCKY', 'O\n'], ['WIN', 'O\n'], [',', 'O\n'], ['CHINA', 'O\n'], ['IN', 'O\n'], ['SURPRISE', 'O\n'], ['DEFEAT', 'O\n'], ['.', 'O\n']]]
    test = addCharInformatioin(sen_list)
    
    predLabels = []
    
    test_set = padding(createMatrices(test, word2Idx, label2Idx, case2Idx,char2Idx))
    
    test_batch,test_batch_len = createBatches(test_set)
    
    for i,data in enumerate(test_batch):    
        tokens, casing,char, labels = data
        #print(tokens, casing,char, labels)
        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, casing,char], verbose=False)[0] 
        pred = pred.argmax(axis=-1) #Predict the classes            
        predLabels.append(pred)
    entity_labels = []
    j = 0
    words_list = sentence.split()
    for i in predLabels[-1]:
        entity_labels.append((words_list[j],idx2Label[int(i)]))
        j+=1
    print("predLabels",entity_labels)    
    
    return entity_labels

getModel()
sentence ="a flight from BLR to MAA on 2018/07/30"
sentence ="SOCCER - JAPAN GET LUCKY WIN CHINA IN SURPRISE DEFEAT."
predict(sentence,entity_extract_model_BLSTM)