from collections import Counter
from tensorflow.contrib import predictor
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

def get_tags(predict_fn,test_data):
    tags = []
    for context in tqdm(test_data):
        ret = predict_fn({"tokens":[context],"length":[len(context)]})
        tag = [item.decode('utf-8') for item in ret['tags'][0]]
        tags.append(tag)
    return tags

def get_format_data(submit_data):
    res = []
    for item in submit_data:
        tmp = []
        context = item['string']
        entities = item['entities']
        idx = 0
        for entity in entities:
            start = entity['start']
            end = entity['end']
            type = entity['type'].lower()
            tmp.append((context[idx:start],'o'))
            tmp.append((context[start:end],type))
            idx = end
        tmp.append((context[idx:],'o'))
        tmp = [t for t in tmp if t[0]]
        res.append(tmp)
    return res

def get_final_tag_result(tag_results, num):
    entities = []
    for tag_res in tag_results:
        entities.extend(tag_res['entities'])

    match_dict = {}
    entities_repr = []
    for entity in entities:
        match_dict[entity.__repr__()] = entity
        entities_repr.append(entity.__repr__())

    final_tags = []
    counter = Counter(entities_repr)
    for key, count in counter.items():
        if count > num / 2:
            final_tags.append(match_dict[key])

    return final_tags

def output_res(output_file, res):
    for sentence in res:
        line = ''
        for segment, tag in sentence:
            line = line + '_'.join(segment) + '/' + tag + '  '
        line = line.strip() + '\n'
        output_file.write(line)
    output_file.flush()

def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    type = ""
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "B":
            if entity_name:
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": type})
            entity_name = ""
            entity_name += char
            entity_start = idx
            type = tag[2:]
        elif tag[0] == "I":
            entity_name += '_'+char
            type = tag[2:]
        else:
            if entity_name:
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": type})
            entity_name = ""
            type = ""
            entity_start = idx
        idx += 1
    return item

if __name__ == '__main__':
    model0 = '/data/public/yinghua/code/OpenNMT-tf/ckpt/daguan_k_fold/0/export/latest/1563783009'
    model1 = '/data/public/yinghua/code/OpenNMT-tf/ckpt/daguan_k_fold/1/export/latest/1563782962'
    model2 = '/data/public/yinghua/code/OpenNMT-tf/ckpt/daguan_k_fold/2/export/latest/1563787680'
    model3 = '/data/public/yinghua/code/OpenNMT-tf/ckpt/daguan_k_fold/3/export/latest/1563782973'
    model4 = '/data/public/yinghua/code/OpenNMT-tf/ckpt/daguan_k_fold/4/export/latest/1563788210'
    models = [model0, model1, model2, model3, model4]

    test_file = 'NERData/raw_data/test.txt'
    test_data = []
    for line in open(test_file, 'r'):
        line = line.strip()
        test_data.append(line.split('_'))

    results = []
    for model_path in models:
        predict_fn = predictor.from_saved_model(model_path)
        tags = get_tags(predict_fn, test_data)
        results.append(tags)

    submit_data = []
    for i in range(len(results[0])):
        tag_results = []
        for tags in results:
            tag_res = result_to_json(test_data[i], tags[i])
            tag_results.append(tag_res)
        tmp = {}
        tmp['string'] = test_data[i]
        tmp['entities'] = get_final_tag_result(tag_results)
        submit_data.append(tmp)

    res = get_format_data(submit_data)
    output_file = open('/data/public/yinghua/data/daguan/most_common_slot_refresh.txt','w')
    output_res(output_file, res)