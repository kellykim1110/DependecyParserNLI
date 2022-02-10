def change_tag(change_dic, root, tag_list, text):
    doc = nlp(text)
    text = [token.text for token in doc]

    if len(change_dic)==0: return root, tag_list, {}

    new_tag_list = [[] for tag in tag_list]
    # print(change_dic)
    for i, tag in enumerate(tag_list):
        for t in tag:
            for k in change_dic.keys():
                if (" ".join(text[min(t[2][0]): 1+ max(t[2][0])]).replace("' ", "'") == k.replace("' ", "'")) and (set(t[2][0]).intersection(change_dic[k]) == set(t[2][0])):
                    t[2][0] = change_dic[k]
                    t[0][0] = " ".join(text[min(t[2][0]): 1+ max(t[2][0])]).replace("' ", "'")
                    if root[0].replace("' ", "'") == k.replace("' ", "'"):
                        root[0] = " ".join(text[min(t[2][0]): 1+ max(t[2][0])])
                if (" ".join(text[min(t[2][1]): 1+ max(t[2][1])]).replace("' ", "'") == k.replace("' ", "'")) and (set(t[2][1]).intersection(change_dic[k]) == set(t[2][1])):
                    t[2][1] = change_dic[k]
                    t[0][1] = " ".join(text[min(t[2][1]): 1 + max(t[2][1])]).replace("' ", "'")
                    if root[0].replace("' ", "'") == k.replace("' ", "'"):
                        root[0] = " ".join(text[min(t[2][1]): 1+ max(t[2][1])]).replace("' ", "'")
                if t[2][0] != t[2][1]:
                    if t not in new_tag_list[i]:
                        new_tag_list[i].append(t)
                #print(t)

    return root, new_tag_list, {}


def merge_tag(texts):
    # {"origin": text, dp:root, words:[[word, tag, idx], [], ... ]}
    #print("\n--------------------------------------------------------------------")
    text = texts["origin"]
    #print(text)
    #print("--------------------------------------------------------------------")

    mod = []; aux = []; comp = []; det = []; other = [];
    mod_tag = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'meta', 'neg', 'nmod', 'npadvmod', 'nummod', 'poss', 'prep', 'quantmod', 'relcl']
    aux_tag = ['aux', 'auxpass']
    comp_tag = ['acomp', 'ccomp', 'pcomp', 'xcomp']
    conj_tag = ["conj"] #'cc', 'preconj', "conj"]
    det_tag = ["det"]

    # 전처리
    dp = "parsing"
    #print(texts[dp]["words"])

    # print("compound일 경우, 하나로 합치기")
    change_dic = {}
    for i, word_pair in enumerate(texts[dp]["words"]):
        tag = word_pair[1][0]
        if tag == "compound":
            if abs(word_pair[2][0][0] - word_pair[2][1][0]) == 1:
                change_dic.update({word_pair[0][0]: word_pair[2][0] + word_pair[2][1]})
                change_dic.update({word_pair[0][1]: word_pair[2][0] + word_pair[2][1]})
    root, tag_list, _ = change_tag(change_dic, texts[dp]["root"], [texts[dp]["words"]], text)
    texts[dp]["root"] = root
    texts[dp]["words"] = tag_list[0]


    # print("conj일 경우 tag 정보 추가")
    conj = {}
    new_word_pair = []
    for i, word_pair in enumerate(texts[dp]["words"]):
        tag = word_pair[1][0]
        if tag == "conj":
            conj[word_pair[2][0][0]] = [word_pair[0][1],word_pair[2][1]]
            conj[word_pair[2][1][0]] = [word_pair[0][0],word_pair[2][0]]
        else: new_word_pair.append(word_pair)

    texts[dp]["words"] = new_word_pair
    del new_word_pair

    new_word_pair = []
    for i, word_pair in enumerate(texts[dp]["words"]):
        if word_pair[2][0][0] in conj.keys():
            if word_pair[1][0] not in mod_tag+aux_tag+comp_tag+det_tag:
                new_word_pair.append([[conj[word_pair[2][0][0]][0], word_pair[0][1]],
                                           word_pair[1],
                                           [conj[word_pair[2][0][0]][1], word_pair[2][1]]])
        elif word_pair[2][1][0] in conj.keys():
            if word_pair[1][0] not in mod_tag + aux_tag + comp_tag + det_tag:
                new_word_pair.append([[word_pair[0][0], conj[word_pair[2][1][0]][0]],
                                           word_pair[1],
                                           [word_pair[2][0], conj[word_pair[2][1][0]][1]]])
    texts[dp]["words"] += new_word_pair
    del new_word_pair


    for i, word_pair in enumerate(texts[dp]["words"]):
        # print(word_pair)
        tag = word_pair[1][0]
        word_idx = word_pair[2]
        if tag in mod_tag: mod.append(word_pair)
        elif tag in aux_tag: aux.append(word_pair)
        elif tag in comp_tag: comp.append(word_pair)
        elif tag in det_tag:det.append(word_pair)
        else: other.append(word_pair)

    # 1
    #using_tag_list = mod + aux + comp
    #tag_list = [x for x in [using_tag_list, det, other] if len(x) != 0]

    # 2
    using_tag_list = mod + aux + comp + det
    tag_list = [x for x in [using_tag_list, other] if len(x) != 0]

    if len(using_tag_list) == 0: return texts

    # len(using_tag_list) != 0
    change_dic = {}
    conti = True
    while conti:
        root, tag_list, change_dic = change_tag(change_dic, texts[dp]["root"], tag_list, text)
        texts[dp]["root"] = root
        tag_li = tag_list[0] #using_tag_list
        #print("tag_li: " + str(tag_li))
        one_tag_li = []
        two_tag_li = []
        other_tag_li = []
        for tag_l in tag_li:
            if abs(max(tag_l[2][1])-min(tag_l[2][0]))==1:
                if tag_l not in one_tag_li: one_tag_li.append(tag_l)
            if abs(max(tag_l[2][1]) - min(tag_l[2][0])) == 2:
                if tag_l not in two_tag_li: two_tag_li.append(tag_l)
            else:
                if tag_l not in other_tag_li: other_tag_li.append(tag_l)

        one_tag_li = sorted(one_tag_li, key=lambda x: (min(x[2][1])))
        two_tag_li = sorted(two_tag_li, key=lambda x: (min(x[2][1])))
        len12 = len(one_tag_li) + len(two_tag_li)

        if (len(tag_li)==1): conti = False
        elif (len(one_tag_li)<2) and (len(two_tag_li)==0): conti = False
        else:
            tag_li = other_tag_li
            del other_tag_li

            # 거리의 길이가 2인 경우
            # print("거리의 길이가 2인 경우")
            j = 0
            while j < len(two_tag_li):
                two_tag = two_tag_li[j]
                i = 0
                before_change_dic = {}
                while i < len(one_tag_li):
                    one_tag = one_tag_li[i]
                    if (max(two_tag[2][1]) == max(one_tag[2][1])) or (
                            max(two_tag[2][1]) == max(one_tag[2][0])) or (max(two_tag[2][0]) == max(one_tag[2][0])):
                        change_dic.update({one_tag[0][0]: list(set(one_tag[2][0] + one_tag[2][1] + two_tag[2][0] + two_tag[2][1]))})
                        change_dic.update({one_tag[0][1]: list(set(one_tag[2][0] + one_tag[2][1] + two_tag[2][0] + two_tag[2][1]))})
                        change_dic.update({two_tag[0][1]: list(set(one_tag[2][0] + one_tag[2][1] + two_tag[2][0] + two_tag[2][1]))})
                        change_dic.update(
                            {two_tag[0][0]: list(set(one_tag[2][0] + one_tag[2][1] + two_tag[2][0] + two_tag[2][1]))})
                    ### (a,c) (b,c) -> a b c
                    #if max(two_tag[2][1]) == max(one_tag[2][1]):
                    #    #print("1")
                    #    change_dic.update({one_tag[0][0]: list(set(one_tag[2][0] + one_tag[2][1] + two_tag[2][1]))})
                    #    change_dic.update({one_tag[0][1]: list(set(one_tag[2][0] + one_tag[2][1] + two_tag[2][1]))})
                    #    change_dic.update({two_tag[0][1]: list(set(one_tag[2][0] + one_tag[2][1] + two_tag[2][1]))})
                    ### (a,b) (b,c) -> a c b
                    #elif max(two_tag[2][1]) == max(one_tag[2][0]):
                    #    #print("2")
                    #    change_dic.update({one_tag[0][0]: list(set(one_tag[2][0] + one_tag[2][1] + two_tag[2][1]))})
                    #    change_dic.update({one_tag[0][1]: list(set(one_tag[2][0] + one_tag[2][1] + two_tag[2][1]))})
                    #    change_dic.update({two_tag[0][1]: list(set(one_tag[2][0] + one_tag[2][1] + two_tag[2][1]))})
                    ### (a,b) (a,c) -> a b c
                    #elif max(two_tag[2][0]) == max(one_tag[2][0]):
                    #    #print("3")
                    #    change_dic.update({one_tag[0][0]: list(set(one_tag[2][0] + one_tag[2][1] + two_tag[2][0]))})
                    #    change_dic.update({one_tag[0][1]: list(set(one_tag[2][0] + one_tag[2][1] + two_tag[2][0]))})
                    #    change_dic.update({two_tag[0][0]: list(set(one_tag[2][0] + one_tag[2][1] + two_tag[2][0]))})


                    if len(change_dic) == 0: i += 1
                    else:
                        if before_change_dic != change_dic:
                            before_change_dic = change_dic
                            root, tag_list, _ = change_tag(change_dic, texts[dp]["root"], tag_list, text)
                            texts[dp]["root"] = root
                            root, one_two_tag_list, change_dic = change_tag(change_dic, texts[dp]["root"],[one_tag_li, two_tag_li], text)
                            texts[dp]["root"] = root
                            one_tag_li = one_two_tag_list[0]
                            two_tag_li = one_two_tag_list[1]
                            one_tag_li = sorted(one_tag_li, key=lambda x: (min(x[2][1])))
                            two_tag_li = sorted(two_tag_li, key=lambda x: (min(x[2][1])))
                        else: i += 1
                j += 1

            # 거리의 길이가 1일 때 양 옆에 이어서 있는 경우
            # print("거리의 길이가 1일 때 양 옆에 이어서 있는 경우")
            one_tag_li = sorted(one_tag_li, key=lambda x: (min(x[2][1])))
            i = 1
            before_change_dic = {}
            while i < len(one_tag_li):
                front_tag = one_tag_li[i-1]
                next_tag = one_tag_li[i]
                if max(front_tag[2][1]) == max(next_tag[2][0]):
                    change_dic.update({next_tag[0][0]: list(set(next_tag[2][0] + next_tag[2][1] + front_tag[2][0] + front_tag[2][1]))})
                    change_dic.update({next_tag[0][1]: list(set(next_tag[2][0] + next_tag[2][1] + front_tag[2][0] + front_tag[2][1]))})
                    change_dic.update({front_tag[0][0]: list(set(next_tag[2][0] + next_tag[2][1] + front_tag[2][0] + front_tag[2][1]))})
                    change_dic.update({front_tag[0][1]: list(set(next_tag[2][0] + next_tag[2][1] + front_tag[2][0] + front_tag[2][1]))})
                ### (a,b) (b,c) -> a b c
                #if max(front_tag[2][1]) == max(next_tag[2][0]):
                #    #print("1")
                #    change_dic.update({next_tag[0][0]:list(set(next_tag[2][0] + next_tag[2][1] + front_tag[2][1]))})
                #    change_dic.update({next_tag[0][1]:list(set(next_tag[2][0] + next_tag[2][1] + front_tag[2][1]))})
                #    change_dic.update({front_tag[0][1]:list(set(next_tag[2][0] + next_tag[2][1] + front_tag[2][1]))})

                ### (a,b) (b,c) -> a b c
                #elif max(front_tag[2][1]) == max(next_tag[2][1]):
                #    #print("2")
                #    change_dic.update({next_tag[0][0]: list(set(next_tag[2][0] + next_tag[2][1] + front_tag[2][1]))})
                #    change_dic.update({next_tag[0][1]: list(set(next_tag[2][0] + next_tag[2][1] + front_tag[2][1]))})
                #    change_dic.update({front_tag[0][1]: list(set(next_tag[2][0] + next_tag[2][1] + front_tag[2][1]))})
                ### (a,b) (b,c) -> a b c
                #elif max(front_tag[2][0]) == max(next_tag[2][0]):
                #    #print("3")
                #    change_dic.update({next_tag[0][0]: list(set(next_tag[2][0] + next_tag[2][1] + front_tag[2][0]))})
                #    change_dic.update({next_tag[0][1]: list(set(next_tag[2][0] + next_tag[2][1] + front_tag[2][0]))})
                #    change_dic.update({front_tag[0][0]: list(set(next_tag[2][0] + next_tag[2][1] + front_tag[2][0]))})

                if len(change_dic) == 0:
                    i += 1
                else:
                    if before_change_dic != change_dic:
                        root, tag_list, _ = change_tag(change_dic, texts[dp]["root"], tag_list, text)
                        texts[dp]["root"] = root
                        root, one_tag_list, change_dic = change_tag(change_dic, texts[dp]["root"], [one_tag_li], text)
                        texts[dp]["root"] = root
                        one_tag_li = one_tag_list[0]
                        one_tag_li = sorted(one_tag_li, key=lambda x: (min(x[2][1])))
                    else: i += 1
            if len12 == (len(one_tag_li) + len(two_tag_li)): conti = False


    if len(det) != 0:
        #print("det")
        for d in det:
            if abs(max(d[2][0])-min(d[2][1]))==1:
                change_dic.update({d[0][0]: list(set(d[2][0]+d[2][1]))})
                change_dic.update({d[0][1]: list(set(d[2][0]+d[2][1]))})
                root, tag_list, _ = change_tag(change_dic, texts[dp]["root"], tag_list, text)
                texts[dp]["root"] = root

    new_tag_list = []

    doc = nlp(text)
    text = [token.text for token in doc]

    merge_text_min = set()
    #print(tag_list)
    for tag in tag_list:
        for t in tag:
            if (t[0][0].strip() in [".", "!", "?", ","]) and (len(t[2][0]) != 1): t[2][0] = [t[2][0][0]]
            if (t[0][1].strip() in [".", "!", "?", ","]) and (len(t[2][1]) != 1): t[2][1] = [t[2][1][0]]

            if t[2][0] != t[2][1]:
                t[0][0] = " ".join(text[min(t[2][0]): 1+ max(t[2][0])]).replace("' ", "'")
                t[2][0] = list(range(min(t[2][0]), 1+ max(t[2][0])))
                t[0][1] = " ".join(text[min(t[2][1]): 1 + max(t[2][1])]).replace("' ", "'")
                t[2][1] = list(range(min(t[2][1]), 1 + max(t[2][1])))
                merge_text_min.add(min(t[2][0]))
                merge_text_min.add(min(t[2][1]))
                if t not in new_tag_list: new_tag_list.append(t)
    texts["merge"] = {"origin": [], "root": texts[dp]["root"]}

    merge_text_min = sorted(list(merge_text_min))
    for i, idxs in enumerate(merge_text_min[:-1]):
        texts["merge"]["origin"].append((" ".join(text[idxs: merge_text_min[i+1]]).replace("' ", "'"), list(range(idxs, merge_text_min[i+1]))))

    # 문장이 다 수식을 해주는 관계인 구형태일 경우 문장 하나가 통으로 청크될 수 있음
    if len(merge_text_min) == 0:
        ## tag_list[i]: 각 tag별 청크간의 관계 리스트, tag_list[i][j]: 각 tag별 청크간의 관계
        if (len(tag_list) == 1) and (len(tag_list[0]) == 1):
            texts["merge"]["origin"].append((tag_list[0][0][0][0], tag_list[0][0][2][0]))
        else: print(tag_list)
    else:
        if len(" ".join(text[merge_text_min[-1]:]).replace("' ", "'")) != 1:
            texts["merge"]["origin"].append(
            (" ".join(text[merge_text_min[-1]:]).replace("' ", "'"), list(range(merge_text_min[-1], 1+len(text)))))
        else:
            texts["merge"]["origin"].append(
                (" ".join(text[merge_text_min[-1]:]).replace("' ", "'"), [merge_text_min[-1]]))
    texts["merge"][dp] = new_tag_list

    del texts[dp]

    return texts

if __name__ == '__main__':
    dir = "./../../data/snli"
    files = ["snli_1.0_test.jsonl", "snli_1.0_dev.jsonl", "snli_1.0_train.jsonl"]

    import spacy
    nlp = spacy.load('en_core_web_sm')

    import json, os
    from tqdm import tqdm
    for file in files:
        final = []
        with open(os.path.join(dir,file), "r", encoding="utf8") as inf:
            for num, line in tqdm(enumerate(inf)):
                data = json.loads(line)
                output = {"guid": file.split(".jsonl")[0]+"_"+"0"*(4-len(str(num)))+str(num),
                          "premise":{"origin": data["sentence1"]},
                          "hypothesis":{"origin":data["sentence2"]},
                          "annotator_labels": data["annotator_labels"],
                          "gold_label": data["gold_label"].strip()}
                for key in ["premise","hypothesis"]:
                    texts = output[key]
                    # texts = {"origin": "The quick brown fox jumps over the lazy dog."}
                    # texts = {"origin": "A guy with a brown shirt and glasses is telling his friend in the glasses and gray shirt a story."}
                    #print(texts["origin"])
                    texts["origin"] = texts["origin"].strip().replace("-", " ").replace("_", " ").replace('"', "").replace("  ", " ")
                    texts["parsing"] = {"root": []}
                    texts["parsing"]["words"] = []

                    doc = nlp(texts["origin"])
                    #for token in doc:
                    #    print("{2}({3}-{6}, {0}-{5})".format(
                    #        token.text, token.tag_, token.dep_, token.head.text, token.head.tag_, token.i+1, token.head.i+1))
                    texts["parsing"]["words"] = [[[token.text, token.head.text],[token.dep_], [[(token.i+1)-1], [(token.head.i+1)-1]]] for token in doc if token.dep_ != "ROOT"]
                    texts["parsing"]["root"] = [token.text for token in doc if token.dep_ == "ROOT"]

                    parsing = texts["parsing"]
                    #print(parsing)
                    output_texts = merge_tag(texts)
                    #print(output_texts)

                    output[key] = output_texts
                final.append(output)

        outfile = file.split(".jsonl")[0]+"_2.json"
        if not os.path.isdir(os.path.join(dir,"parsing")):
            os.mkdir(os.path.join(dir,"parsing"))
        with open(os.path.join(dir,"parsing",outfile), "w", encoding="utf8") as outf:
            json.dump(final, outf, indent=2)
        print("does the file exist?: "+str(os.path.isfile(os.path.join(dir,"parsing",outfile))))