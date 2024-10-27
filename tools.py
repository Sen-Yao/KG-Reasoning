import torch
from util import getDistance

def getTemplate(args, data):
    edge = data['edge'][:-1] if len(data['edge'])<=(args.len_arg)//10 else data['edge'][0:(args.len_arg)//10]
    causeRatio = args.cause_ratio/(args.cause_ratio+args.becausedby_ratio)
    causeRel = edge[0:int(len(edge)*causeRatio)]
    becausedbyRel = [[rel[-1], 'be casued by', rel[0]] for rel in edge[int(len(edge)*causeRatio):]]
    template = ''
    relation = [] + causeRel + becausedbyRel
    for rel in relation:
        eId1 = rel[0]
        eId2 = rel[-1]
        rl = data['node'][eId1][5] + ' ' + rel[1] + ' ' + data['node'][eId2][5]
        template = template + rl + ' , '
    maskRel = data['edge'][-1]
    return template + data['node'][maskRel[0]][5] + ' ' + maskRel[1] + ' <mask> .', relation + [maskRel]

def getTemplate_SeDGPL(args, data):
    edge = data['edge'][:-1] if len(data['edge'])<=(args.len_arg)//10 else data['edge'][0:(args.len_arg)//10]
    # random.shuffle(edge)
    causeRel = edge[0:len(edge)]
    template, templateType = '', ''
    relation = [] + causeRel
    assert data['edge'][-1] not in relation

    distance = getDistance(relation+[data['edge'][-1]])
    assert len(relation)+1==len(distance)

    weighted_characters = list(zip(distance[:-1], relation))
    sorted_characters = sorted(weighted_characters, reverse=True)
    sorted_relation_only = [char for weight, char in sorted_characters]


    # random.shuffle(relation)
    for rel in sorted_relation_only:
        eId1 = rel[0]
        eId2 = rel[-1]
        rl = data['node'][eId1][5] + ' ' + rel[1] + ' ' + data['node'][eId2][5]
        rlType = data['node'][eId1][4] + ' '+rel[1]+' ' + data['node'][eId2][4]
        template = template + rl + ' , '
        templateType = templateType + rlType + ' , '
    maskRel = data['edge'][-1]
    template = template + data['node'][maskRel[0]][5] + ' ' + maskRel[1] + ' <mask> .'
    templateType = templateType + data['node'][maskRel[0]][4] + ' '+maskRel[1]+' <mask> .'
    assert len(template.split(' ')) == len(templateType.split(' '))
    return template, templateType, sorted_relation_only + [maskRel]

def getSentence(args, tokenizer, data, relation):
    sentence = {}
    for rel in relation:
        # 对于一条 rel 而言，其起始和结束节点可能是待预测的节点，而待预测节点在 data 中为空，因此这里取句子时需要注意
        # 若起始节点不在字典中
        if rel[0] not in sentence.keys() and rel[0] != len(data['node'])-1:
            # 起始节点为无效的待预测节点
            if rel[0] == len(data['node'])-1:
                sentence[rel[0]] = data['node'][rel[-1]][6]
            else:
                sentence[rel[0]] = data['node'][rel[0]][6]
        
        if rel[-1] not in sentence.keys() and rel[-1] != len(data['node'])-1:
            # 起始节点为无效的待预测节点
            if rel[-1] == len(data['node'])-1:
                sentence[rel[-1]] = data['node'][rel[0]][6]
            else:
                sentence[rel[-1]] = data['node'][rel[-1]][6]
    sentTokenizer = {}
    for e in sentence.keys():
        sent_dict = tokenizer.encode_plus(
                sentence[e],
                add_special_tokens=True,
                padding='max_length',
                max_length=args.len_arg,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        event = data['node'][e][5]

        sentTokenizer[str(e)+'_'+str(tokenizer.encode(event)[1])] = {'input_ids':      sent_dict['input_ids'],
                                                                     'attention_mask': sent_dict['attention_mask'],
                                                                     'position':       torch.where(sent_dict['input_ids']==tokenizer.encode(event)[1])[1].item()}
    return sentTokenizer

def tokenizerHandler(args, template, tokenizer):
    encode_dict = tokenizer.encode_plus(
        template,
        add_special_tokens=True,
        padding='max_length',
        max_length=args.len_arg,
        truncation=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    arg_1_idx = encode_dict['input_ids']
    arg_1_mask = encode_dict['attention_mask']
    return arg_1_idx, arg_1_mask

def getposHandler(data, arg_idx, relation, sentence, tokenizer):
    tempPosition = torch.nonzero(arg_idx >= tokenizer.encode('<a_0>')[1]).tolist()
    ePosition = [row[-1] for row in tempPosition]
    ePositionKey = []
    sentId = []
    for rel in relation:
        sentId.append(rel[0])
        sentId.append(rel[-1])
    assert len(sentId) - 1 == len(ePosition)
    for iid in range(len(ePosition)):
        event = data['node'][sentId[iid]][5]
        ePositionKey.append(str(sentId[iid]) + '_' + str(arg_idx[0][ePosition[iid]].item()))
    return ePosition, ePositionKey

# tokenize sentence and get event idx
def get_batch(data, args, indices, tokenizer, with_labels=True):
    batch_idx = []
    batch_mask = []
    mask_indices = []   # mask所在位置
    labels = []         # 存储真实事件标签的id
    candiSet = []  # 存储候选节点的id
    for idx in indices:
        candi = [tokenizer.encode(data[idx]['candiSet'][i])[1] for i in range(len(data[idx]['candiSet']))]
        template, relation = getTemplate(args, data[idx])

        encode_dict = tokenizer.encode_plus(
            template,
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        arg_1_idx = encode_dict['input_ids']
        arg_1_mask = encode_dict['attention_mask']
        if with_labels:
            label = tokenizer.encode(data[idx]['candiSet'][data[idx]['label']])[1]
            labels.append(label)
        candiSet.append(candi)

        if len(batch_idx) == 0:
            batch_idx = arg_1_idx
            batch_mask = arg_1_mask
            mask_indices = torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1]
            mask_indices = torch.unsqueeze(mask_indices, 0)
        else:
            batch_idx = torch.cat((batch_idx, arg_1_idx), dim=0)
            batch_mask = torch.cat((batch_mask, arg_1_mask), dim=0)
            mask_indices = torch.cat((mask_indices, torch.unsqueeze(torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1], 0)), dim=0)
    if with_labels:
        return batch_idx, batch_mask, mask_indices, labels, candiSet
    else:
        return batch_idx, batch_mask, mask_indices, candiSet
    

# tokenize sentence and get event idx
def get_batch_SeDGPL(data, args, indices, tokenizer, with_label=True):
    batch_idx, batch_mask = [], []
    batch_Type_idx, batch_Type_mask = [], []
    event_tokenizer_pos, event_key_pos = [], []
    mask_indices, sentences, labels, candiSet= [],[],[],[]
    for idx in indices:
        candi = [tokenizer.encode(data[idx]['candiSet'][i])[1] for i in range(len(data[idx]['candiSet']))]
        template, templateType, relation = getTemplate_SeDGPL(args, data[idx])
        sentence = getSentence(args, tokenizer, data[idx], relation)

        arg_idx, arg_mask = tokenizerHandler(args, template, tokenizer)
        # arg_Type_idx, arg_Type_mask = tokenizerHandler(args, templateType, tokenizer)

        # assert arg_mask.tolist() == arg_Type_mask.tolist()
        assert relation[-1] == data[idx]['edge'][-1]
        
        if with_label:
            # assert candi[data[idx]['label']] == tokenizer.encode(data[idx]['node'][relation[-1][-1]][5])[1]
            # label = tokenizer.encode(data[idx]['node'][relation[-1][-1]][5])[1]
            label = candi[data[idx]['label']]
            labels.append(label)
        # template分词后所有事件的位置
        ePosition, ePositionKey = getposHandler(data[idx], arg_idx, relation, sentence, tokenizer)
        # eTypePosition, eTypePositionKey = getposHandler(data[idx], arg_Type_idx, relation, sentence, tokenizer)
        # assert ePosition == eTypePosition
        event_tokenizer_pos.append(ePosition)
        event_key_pos.append(ePositionKey)
        sentences.append(sentence)
        candiSet.append(candi)
        if len(batch_idx) == 0:
            batch_idx, batch_mask = arg_idx, arg_mask
            # batch_Type_idx, batch_Type_mask = arg_Type_idx, arg_Type_mask
            mask_indices = torch.nonzero(arg_idx == 50264, as_tuple=False)[0][1]
            mask_indices = torch.unsqueeze(mask_indices, 0)
        else:
            batch_idx, batch_mask = torch.cat((batch_idx, arg_idx), dim=0), torch.cat((batch_mask, arg_mask), dim=0)
            # batch_Type_idx, batch_Type_mask = torch.cat((batch_Type_idx, arg_Type_idx), dim=0), torch.cat((batch_Type_mask, arg_Type_mask), dim=0)
            mask_indices = torch.cat((mask_indices, torch.unsqueeze(torch.nonzero(arg_idx == 50264, as_tuple=False)[0][1], 0)), dim=0)
    if with_label:
        return batch_idx, batch_mask, batch_Type_idx, batch_Type_mask, event_tokenizer_pos, event_key_pos, mask_indices, sentences, labels, candiSet
    else:
        return batch_idx, batch_mask, batch_Type_idx, batch_Type_mask, event_tokenizer_pos, event_key_pos, mask_indices, sentences, candiSet



# calculate Hit@1 Hit@3 Hit@10
def calculate(prediction, candiSet, labels, batch_size):
    '''
    prediction: 一个 batch 的每个样本的预测结果，二维张量
    candiSet: 一个 batch 的每个样本候选集下标指示，二维张量
    labels: 一个 batch 的每个样本的 GT，二维张量
    '''
    hit1, hit3, hit10, hit50 = [], [], [], []
    for i in range(batch_size):
        # predtCandi 是一个一维数组，表示对当前样本来说，所有的此样本候选事件的发生概率
        predtCandi = prediction[i][candiSet[i]].tolist()
        # 根据 gt，在候选集中的第 label 个事件是真实发生的
        label = candiSet[i].index(labels[i])
        # 模型认为 gt 发生的概率
        labelScore = predtCandi[label]
        # 对分数从高到低排序
        predtCandi.sort(reverse=True)
        # 模型认为 gt 的排名
        rank = predtCandi.index(labelScore)
        hit1.append(int(rank<1))
        hit3.append(int(rank<3))
        hit10.append(int(rank<10))
        hit50.append(int(rank < 50))

    return hit1, hit3, hit10, hit50


def isContinue(id_list):
    for i in range(len(id_list) - 1):
        if int(id_list[i]) != int(id_list[i + 1]) - 1:
            return False
    return True


def doCorrect(data):
    for i in range(len(data)):
        eId = data[i][8].split('_')[1:]
        if not isContinue(eId):
            s_1 = data[i][6].split()
            event1 = s_1[int(eId[0]):int(eId[-1]) + 1]
            event1 = ' '.join(event1)
            event1 += ' '  # 在这里加空格是 因为其它event后面都有一个空格，这里仅仅是为了与它们保持一致
            new_e1_id = [str(i) for i in range(int(eId[0]), int(eId[-1]) + 1)]
            temp = ''
            for ii in new_e1_id:
                temp += s_1[int(ii)] + ' '
            assert event1 == temp
            event_place1 = '_' + '_'.join(new_e1_id)
            sentence = (
            data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], event1, data[i][6], data[i][7],
            event_place1)
            data[i] = sentence
    return data

def correct_data(dataSet):
    for i in range(len(dataSet)):
        dataSet[i]['node'] = doCorrect(dataSet[i]['node'])
        # dataSet[i]['candiSet'] = doCorrect(dataSet[i]['candiSet'])
    return dataSet

def doCollect(data, tokenizer, multi_event, to_add, special_multi_event_token, event_dict, reverse_event_dict, flag_candi=0):
    for sentence in data:
        if flag_candi == 0:
            event = sentence[5]
        else:
            event = sentence
        # 为了方便后续替换，这里选择把所有的事件都进行替换
        if event not in multi_event:
            multi_event.append(event)
            special_multi_event_token.append("<a_" + str(len(special_multi_event_token)) + ">")
            event_dict[special_multi_event_token[-1]] = multi_event[-1]
            reverse_event_dict[multi_event[-1]] = special_multi_event_token[-1]
            to_add[special_multi_event_token[-1]] = tokenizer(multi_event[-1].strip())['input_ids'][1: -1]
    return multi_event, to_add, special_multi_event_token, event_dict, reverse_event_dict


def collect_mult_event(train_data, tokenizer):
    multi_event = []
    to_add = {}
    special_multi_event_token = []
    event_dict = {}
    reverse_event_dict = {}
    for sentence in train_data:
        multi_event, to_add, special_multi_event_token, event_dict, reverse_event_dict = doCollect(sentence['node'][:-1],
                                                                                                   tokenizer,
                                                                                                   multi_event, to_add,
                                                                                                   special_multi_event_token,
                                                                                                   event_dict,
                                                                                                   reverse_event_dict)
        multi_event, to_add, special_multi_event_token, event_dict, reverse_event_dict = doCollect(
                                                                                                   sentence['candiSet'][:-1],
                                                                                                   tokenizer,
                                                                                                   multi_event, to_add,
                                                                                                   special_multi_event_token,
                                                                                                   event_dict,
                                                                                                   reverse_event_dict,1)
    return multi_event, special_multi_event_token, event_dict, reverse_event_dict, to_add


def doReplace(data, reverse_event_dict):
    for i in range(len(data)):
        # assert data[i][5] in reverse_event_dict
        if data[i][5] in reverse_event_dict:
            sent = data[i][6].split()
            eId = data[i][8].split('_')[1:]
            eId.reverse()
            for id in eId:
                sent.pop(int(id))
            sent.insert(int(eId[-1]), reverse_event_dict[data[i][5]])
            sentence = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], reverse_event_dict[data[i][5]], " ".join(sent),
                            data[i][7], '_' + eId[-1])
            data[i]=sentence
    return data

def replace_mult_event(data, reverse_event_dict):
    for i in range(len(data)):
        data[i]['node'] = doReplace(data[i]['node'], reverse_event_dict)
        temp = [reverse_event_dict[e] for e in data[i]['candiSet']]
        data[i]['candiSet'] = temp
    return data






if __name__ == '__main__':
    pass
