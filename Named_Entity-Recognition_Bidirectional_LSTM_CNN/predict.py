from preprocessing import createBatches,createMatrices,addCharInformatioin,padding
from keras.models import load_model
import pickle


print("Model is loading...")
with open("Benchmark_Models/dict/wd_to_id.txt", "rb") as myFile:
    wd_to_id = pickle.load(myFile)

with open("Benchmark_Models/dict/lb_to_id.txt", "rb") as myFile:
    lb_to_id = pickle.load(myFile)

with open("Benchmark_Models/dict/ch_to_id.txt", "rb") as myFile:
    ch_to_id = pickle.load(myFile)

with open("Benchmark_Models/dict/idx2Label.txt", "rb") as myFile:
    idx2Label = pickle.load(myFile)




#from nn import entity_extract_model_BLSTM
import numpy as np

entity_model = load_model('Benchmark_Models/Benchmark_all_Entity_Model_try_with_book_flight.h5')

print("Completed")


def predict(sentence,model):
    sen_list = [[[i,'O\n'] for i in sentence.split()]]
    test = addCharInformatioin(sen_list)
    
    predLabels = []
    
    test_set = padding(createMatrices(test,wd_to_id,  lb_to_id, ch_to_id))
    
    test_batch,test_batch_len = createBatches(test_set)
    for i,data in enumerate(test_batch):
        tokens, char, labels = data
        tokens = np.asarray([tokens])     
        char = np.asarray([char])
        pred = model.predict([tokens,char], verbose=False)[0] 
        pred = pred.argmax(axis=-1) #Predict the classes            
        predLabels.append(pred)
    entity_labels = []
    j = 0
    words_list = sentence.split()
    for i in predLabels[-1]:
        entity_labels.append((words_list[j],idx2Label[int(i)].replace("\n","")))
        j+=1
    
    return entity_labels

#sentence ="a flight from ccu to hyd on 2018/07/30"
#list_sen = ["a flight from ccu to hyd on 2018/07/30","Make a booking from ccu to hyd on 2018/07/30","CCU","hyd"]
test_File = open("Test_Data/Test_benchmark.txt")
test_list = test_File.readlines()

test_list = ['book a flight for Goa to haryana on 2016/11/30']
pred_benchmark_list = []
for i in test_list:
    pred_benchmark_list.append(predict(i,entity_model))

print("Enter/Paste your content. Press double enter to exit.")

contents = []
while True:
    try:
        line = input()
        entity_prediction = predict(line,entity_model)
        for i in entity_prediction:
            print(i)
    except EOFError:
        break
    contents.append(line)
'''    
def make_txt(list):
    
    f = open("Test_Data/benchmark_entity_result.txt",'w')  
    f.write("\n".join(list))
    
make_txt(pred_benchmark_list)
'''
