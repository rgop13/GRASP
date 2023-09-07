import json
import re


def tokenize_vanilla(text, tokenizer, subject=None, object=None, trigger_list=None, prompt_token=None):
    D = ['[unused1]', '[unused2]']
    textraw = [text]
    for delimiter in D:
        ntextraw = []
        for i in range(len(textraw)):
            t = textraw[i].split(delimiter)
            for j in range(len(t)):
                ntextraw += [t[j]]
                if j != len(t) - 1:
                    ntextraw += [delimiter]
        textraw = ntextraw
    text = []
    for t in textraw:
        if t in ['[unused1]', '[unused2]']:
            text += [t]
        else:
            tokens = tokenizer.tokenize(t)
            for tok in tokens:
                text += [tok]
    return text


def APMTokenize(text, tokenizer, subject=None, object=None, trigger_list=None, prompt_token=None):
    D = ['[unused1]', '[unused2]']
    if subject is not None:
        D.append(subject)
    if object is not None:
        D.append(object)
    if trigger_list is not None:
        for trigger in trigger_list:
            D.append(trigger)

    textraw = [text]
    for delimiter in D:
        ntextraw = []
        for i in range(len(textraw)):
            t = textraw[i].split(delimiter)
            for j in range(len(t)):
                ntextraw += [t[j]]
                if j != len(t) - 1:
                    ntextraw += [delimiter]
        textraw = ntextraw
    text = []
    for t in textraw:
        if t in ['[unused1]', '[unused2]']:
            text += [t]
        else:
            tokens = tokenizer.tokenize(t)
            for tok in tokens:
                text += [tok]
    return text


def get_target_mask(input_ids, target_ids, target_string, pad_token_id, tokenizer):
    target_ids_with_space = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + target_string))
    if isinstance(target_ids, list):
        target_size = len(target_ids)
    elif isinstance(target_ids, int):
        target_size = 1
        target_ids = [target_ids]
        target_ids_with_space = [target_ids_with_space]
    else:
        raise ValueError("\'target_ids\' should has List or Integer type!")
    mask = [0 for _ in range(len(input_ids))]
    for i in range(len(input_ids) - target_size + 1):
        if input_ids[i] == pad_token_id:
            break
        if input_ids[i:i + target_size] == target_ids or input_ids[i:i + target_size] == target_ids_with_space:
            mask[i:i + target_size] = [1] * target_size
    return mask


def is_speaker(a):
    if re.match(r'speaker\s?\d+', a):
        return True
    else:
        return False
    # a = a.split()
    # return len(a) == 2 and a[0] == "speaker" and a[1].isdigit()


def rename(d, x, y):
    d = d.replace("’", "'")
    d = d.replace("...", ".")
    d = d.replace("—", "-")
    d = d.replace("…", ".")
    # if "dialogue:berts" in self.args.prompt_type:
    unused = ["[unused1]", "[unused2]"]
    a = []
    if is_speaker(x):
        a += [x]
    else:
        a += [None]
    if x != y and is_speaker(y):
        a += [y]
    else:
        a += [None]
    for i in range(len(a)):
        if a[i] is None:
            continue
        d = d.replace(a[i] + ":", unused[i] + " :")
        if x == a[i]:
            x = unused[i]
        if y == a[i]:
            y = unused[i]
    return d, x, y


def read_jsonl(file_reader):
    lines = file_reader.readlines()
    data = []
    for line in lines:
        d = json.loads(line)
        data.append(d)
    return data
