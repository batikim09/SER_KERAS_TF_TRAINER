from sklearn.metrics import f1_score,recall_score,confusion_matrix
import numpy as np
import scipy.stats as st

total_cm = {}
total_pred = {}
total_label = {}

def collect_add_cm(task, pred, label):
    task_pred = total_pred.get(task)
    
    pred = pred.tolist()
    label = label.tolist()
    
    if task_pred == None:
        total_pred[task] = pred
    else:
        task_pred.extend(pred)
        
    task_label = total_label.get(task)
    if task_label == None:
        total_label[task] = label
    else:
        task_label.extend(label)

def frame_level_evaluation(model, X, dictForLabelsTest, multiTasks, stl, reg = False, dataset = ''): 
    predictions = model.predict([X])
    id = 0
    scores = []
    for task, classes, idx in multiTasks:

        if reg:
            if stl == True:
                scores.append( ccc(predictions, dictForLabelsTest[task]))    
            else:
                scores.append( ccc(predictions[id], dictForLabelsTest[task]))
        else:
            if stl == True:
                scores.append( unweighted_recall_time(predictions, dictForLabelsTest[task], dataset))    
            else:
                scores.append( unweighted_recall_time(predictions[id], dictForLabelsTest[task], dataset))
        id = id + 1
    return scores

def unweighted_recall_task(predictions, dictForLabelsTest, multiTasks, class_weight_dict = None, dataset = ''):
    count = 0
    #for STL
    scores = []
    if len(multiTasks) == 1:

        labels = np.argmax(list(dictForLabelsTest.values())[0],1)
        pred = np.argmax(predictions,1)
        score = recall_score(labels, pred, average='macro')
        
       
        cm = confusion_matrix(labels, pred)
        prob_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        prob_cm = np.nan_to_num(prob_cm)
        
        task = list(dictForLabelsTest.keys())[0] + '_' + dataset
        
        collect_add_cm(task, pred, labels)
        
        print("cm: ", prob_cm)
        print("single task", task)
        print("unweighted recall: ", score)
        scores.append(score)
        cm_list = total_cm.get(task)
        if cm_list == None:
            total_cm[task] = [prob_cm]
        else:
            cm_list.append(prob_cm)
            
    else:#for MTL
        for task, classes, idx in multiTasks:
            '''
            if class_weight_dict:
                #print ("before weighting output, ", str(predictions[count]))
                print ("weight, ", str(class_weight_dict[task]))
                predictions[count] = class_weight_dict[task] * predictions[count]
                #print ("after weighting output, ", str(predictions[count]))
            '''
            labels = np.argmax(dictForLabelsTest[task],1)
            pred = np.argmax(predictions[count],1)
            score = recall_score(labels, pred, average='macro')
           
            cm = confusion_matrix(labels, pred)
            prob_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            prob_cm = np.nan_to_num(prob_cm)
            
            task += '_' + dataset
            print("task: ", task)
            print("unweighted recall: ", task, ": ", score)
            print("cm: ", prob_cm)

            collect_add_cm(task, pred, labels)

            scores.append(score)
            count = count + 1
            
            cm_list = total_cm.get(task)
            if cm_list == None:
                total_cm[task] = [prob_cm]
            else:
                cm_list.append(prob_cm)

    return scores

def regression_task(predictions, dictForLabelsTest, multiTasks, dataset = '',):
    count = 0
    #for STL
    scores = []
    if len(multiTasks) == 1:
        labels = list(dictForLabelsTest.values())[0]
        pred = predictions
        score = ccc(pred, labels)

        task = list(dictForLabelsTest.keys())[0] + '_' + dataset
        collect_add_cm(task, pred, labels)
        print("ccc: ", score)
        scores.append(score)
    else:#for MTL
        for task, classes, idx in multiTasks:
            labels = dictForLabelsTest[task]
            pred = predictions[count]
            score = ccc(pred, labels)
            print("ccc: ", score)
            scores.append(score)
            count = count + 1
            task += '_' + dataset
            collect_add_cm(task, pred, labels)
    return scores

def ccc(predictions, labels, norm = True, reshape = True):
    print('pred shape: ', predictions.shape )
    print('label shape: ', labels.shape )
    if reshape:
        predictions = np.reshape(predictions, (predictions.shape[0] * predictions.shape[1]))
        print('after reshaping, pred shape: ', predictions.shape )   
    #min-max normalisation
    if norm:
        min_p = np.min(predictions)
        max_p = np.max(predictions)
        predictions = (predictions - min_p) / (max_p - min_p)
        min_l = np.min(labels)
        max_l = np.max(labels)
        labels = (labels - min_l) / (max_l - min_l)
    m_x = np.mean(predictions)
    m_y = np.mean(labels)
    v_x = np.var(predictions)
    v_y = np.var(labels)
    cov_x_y = np.sum((predictions - m_x) * (labels - m_y))/len(predictions)
    #cov_x_y = np.mean(np.cov(predictions,labels).diagonal()), numpy imp. is strange.
    ccc = (2 * cov_x_y) / (v_x + v_y + (m_x - m_y) * (m_x - m_y))
    return ccc

def unweighted_recall(predictions, labels, task = 'main', dataset = 'test'):
    labels = np.argmax(labels, 1)
    pred = np.argmax(predictions, 1)
    print('pred shape: ', pred.shape )
    print('label shape: ', labels.shape )

    score = recall_score(labels, pred, average='macro')
    print("unweighted recall: ", score)

    cm = confusion_matrix(labels, pred)
    prob_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    prob_cm = np.nan_to_num(prob_cm)
    print("cm: ", prob_cm)
    
    task += '_' + dataset
    collect_add_cm(task, pred, labels)

    cm_list = total_cm.get(task)
    if cm_list == None:
        total_cm[task] = [prob_cm]
    else:
        cm_list.append(prob_cm)

    return score

def unweighted_recall_time(predictions, labels, task = 'main', dataset = ''):
    #print('shape of predictions: ', predictions.shape)
    r_predictions = np.reshape(predictions, (predictions.shape[0] * predictions.shape[1], predictions.shape[2]))
    r_labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], labels.shape[2]))
    labels = np.argmax(r_labels, 1)
    pred = np.argmax(r_predictions, 1)
    score = recall_score(labels, pred, average='macro')
    print("unweighted recall: ", score)
    cm = confusion_matrix(labels, pred)
    prob_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    prob_cm = np.nan_to_num(prob_cm)
    print("cm: ", prob_cm)
    
    task += '_' + dataset
    collect_add_cm(task, pred, labels)
    
    cm_list = total_cm.get(task)
    if cm_list == None:
        total_cm[task] = [prob_cm]
    else:
        cm_list.append(prob_cm)

    return score       

def total_write_ccc(test_writer):

    test_writer.write('average ccc\n')
    for task in total_label:
        pred = np.array(total_pred[task])
        labels = np.array(total_label[task])
        score = ccc(pred, labels)
        print(task)
        print(str(score))
        test_writer.write(task + '\n')
        test_writer.write(str(score) + '\n')

def total_write_cm(test_writer):
    print('average over folds: confusion matrix')
    test_writer.write('average over folds: confusion matrix\n')
    #print(str(total_label))
    for task in total_label:
        #print(str(total_pred[task]))
        #print(str(total_label[task]))
        
        pred = np.array(total_pred[task])
        labels = np.array(total_label[task])
        cm = confusion_matrix(labels, pred)
        prob_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        prob_cm = np.nan_to_num(prob_cm)
        unweighted_acc = np.mean(prob_cm.diagonal())
        print("task: ", task)
        print(str(cm))
        print(str(prob_cm))
        print(str(unweighted_acc))
        test_writer.write(task + '\n')
        test_writer.write(str(cm) + '\n')
        test_writer.write(str(prob_cm) + '\n')
        test_writer.write('unweighted acc:\t' + str(unweighted_acc) + '\n')

def total_write_collected_cm(test_writer):
    print('fold-independent confusion matrix')
    test_writer.write('fold-independent confusion matrix\n')
    for task in total_label:
        print("task: ", task)
        print(str(total_cm[task]))
        test_writer.write(task + '\n')
        test_writer.write(str(total_cm[task]) + '\n')
            