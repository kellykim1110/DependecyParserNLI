from koalanlp.Util import initialize, finalize
from koalanlp import API
import json
import re
import pandas as pd
from tqdm import tqdm

# 구문분석
from koalanlp.proc import Parser

# 의존구문분석
def sentences_dependency_parser(morph, text):
    if morph == 'hnn':
        initialize(HNN='2.0.3')
        parser = Parser(API.HNN)

    text = text.strip()
    sentences = parser(text)

    results = []
    for sent in sentences:
        r = []; sub_output = {};
        #print("===== Sentence =====")
        #print(sent.singleLineString())
        #print("Dependency Parse result")

        dependencies = sent.getDependencies()
        if len(dependencies) > 0:
            #print(dependencies)

            for edge in dependencies:
                if edge.getGovernor() is None:
                    sub_output["root"] = edge.getDependent().getSurface()
                else:
                    r.append([[edge.getDependent().getSurface(), edge.getGovernor().getSurface()],
                                                [str(edge.getType()), str(edge.getDepType())]])
        else:
            print("(Unexpected) NULL!")

        sub_output["words"] = r
        results.append(sub_output)
    finalize()

    return sentences, results


# text와 sentence의 어절 정보
#input: text, sentence: list
#output : sen2text: {sentence의 어절 위치: [그에 해당하는 text의 위치들]}
def sen2text(text, sentence):
    # text: origin text
    # text의 어절 정보
    idx2text_word = {};text_word2idx ={sen:[] for sen in text};
    for j, sen in enumerate(text):
        idx2text_word[j] = sen
        text_word2idx[sen].append(j)
    print(idx2text_word)

    # sentence의 어절 정보
    idx2sen_word = {};sen_word2idx  ={sen:[] for sen in sentence};
    for j, sen in enumerate(sentence):
        idx2sen_word[j] = sen
        sen_word2idx[sen].append(j)
    print(idx2sen_word)


    if sentence == text:
        sen2text = {i:[i] for i in range(0, len(text))}
    else:
        sen2text = {i:[] for i in range(0, len(sentence))}
        for i, s  in enumerate(sentence):
            for j,t in enumerate(text):
                if s == t:
                    sen2text[i].append(j)
        print(sen2text)

        max_text = 0
        for i, (s1,s2) in enumerate(zip(list(sen2text.keys())[:-1], list(sen2text.keys())[1:])):
            s1_delete = []
            if i == 0:
                for t in sen2text[s1]:
                    if (t>0) and (small not in sen2text[s1] for small in range(0, t)):
                        s1_delete.append(t)
                    elif (sen2text[s2]!= []) and (t > min(sen2text[s2])):
                        s1_delete.append(t)
                print(s1_delete)
                s1_delete = list(set(s1_delete))
                for t in s1_delete:
                    sen2text[s1].remove(t)
                print("sen2text[s1]: "+str(sen2text[s1]))
                if sen2text[s1] != []: max_text = max(sen2text[s1])
            else:
                if len(sen2text[s1]) > 1:
                    s1_delete = s1_delete + sen2text[s1]
                else:
                    for t in sen2text[s1]:
                        if ((sen2text[list(sen2text.keys())[i - 1]] != []) and (sen2text[s2] != [])):
                            if (max_text > t):
                                s1_delete.append(t)
                            elif (t > min(sen2text[s1])) and (t > min(sen2text[s2])): s1_delete.append(t)
                        elif ((sen2text[list(sen2text.keys())[i - 1]] != []) and (sen2text[s2] == [])):
                            if ((t-max_text)>1) and (small not in sen2text[s1] for small in range(max_text, t)):
                                s1_delete.append(t)
                        elif ((sen2text[list(sen2text.keys())[i - 1]] == []) and (sen2text[s2] != [])):
                            if (max_text > t):
                                s1_delete.append(t)
                            if (max_text > min(sen2text[s2])) and (min(sen2text[s2]) in sen2text[s1]):
                                s1_delete.append(min(sen2text[s2]))
                            else:
                                #if (t > min(sen2text[s2])) and (t in sen2text[s1]): s1_delete.append(t)
                                if (t > min(sen2text[s2])) and (t in sen2text[s2]): s1_delete.append(t)
                print(s1_delete)
                s1_delete = list(set(s1_delete))
                for t in s1_delete:
                    sen2text[s1].remove(t)
                if sen2text[s1] != []: max_text = max(sen2text[s1])
                print("max_text: "+str(max_text))

        # 후처리
        for k,v in sen2text.items():
            v = sorted(v)
            if len(v) > 1:
                for i in range(0,len(v)-1):
                    if v[i]+1!=v[i+1]:sen2text[k]=[]
        print(sen2text)
        # text에는 존재하지만 sentence에 없는 어절
        t_not_s = [i for i, x in enumerate(text) if (i not in sum(list(sen2text.values()),[]))]
        print(t_not_s)
        # # sentence에는 존재하지만 text에 없는 어절
        s_not_t = [i for i, x in enumerate(sentence) if (i in [k for k,v in sen2text.items() if v == []])]
        print(s_not_t)


        """
        중복 어휘는 하나만 존재해도 개수에 상관없이 인식 못할 수 있음
        중복 어휘에 대해서 서로 없는 어절의 양옆에 존재하는지 확인하고 존재한다면 추가
        text: ['지난해', '발표한', '청정', '전남', '블루', '이코노미는', '블루', '에너지,', '블루', '투어,', '블루', '바이오,',
                 '블루', '트랜스포트,', '블루', '농수산,', '블루', '시티', '등', '6개', '분야를', '의미한다.']
        sentence: ['지난해', '발표한', '청정전', '남블루', '이코노미는', '블루에너지,', '블루', '투여,', '블루', '바이오,',
                 '블루', '트랜스포트,', '블루농수산,', '블루시티', '등', '6개', '분야를', '의미한다.']
        """

        t_add=[]; s_add = [];
        more_than = {t:[[],[]] for j,t in enumerate([x for i, x in enumerate(text) if (text.count(x) >= 2)]+[x for i, x in enumerate(sentence) if (sentence.count(x) >= 2)])}
        for i,s in enumerate(sentence):
            if s in more_than.keys():more_than[s][1].append(i)
        for i, s in enumerate(text):
            if s in more_than.keys():more_than[s][0].append(i)
        for k,v in more_than.items():
            if len(v[0]) != len(v[1]):
                t_add += v[0]
                s_add += v[1]
        # print(more_than)
        # print(t_add)
        # print(s_add)

        t_not_s = sorted(list(set(t_not_s+t_add)))
        s_not_t = sorted(list(set(s_not_t+s_add)))
        print(t_not_s)
        print(s_not_t)

        #  t_not_s나 s_not_t에 대하여(각 리스트의 인덱스 번호로 구성)
        # 각 리스트가 서로 연속된 숫자를 기준으로 하나의 말뭉치로 묶여질 가능성이 크다.
        #'블루', '에너지,', '블루', '투어,', '블루', '바이오,', '블루', '트랜스포트,', '블루', '농수산,', '블루', '시티'
        #'블루에너지,', '블루', '투여,', '블루', '바이오,', '블루', '트랜스포트,', '블루농수산,', '블루시티'
        # output: sentence와 text의 말뭉치가 서로 같은 것을 의미하는 것끼리 인덱스 번호로 묶어준다
        #[[블루, 에너지][블루에너지] [] ...]
        re_t_not_s = t_not_s.copy()
        re_s_not_t = s_not_t.copy()
        outputs = []
        sub1 = [];sub2 = [];
        i = 0
        while (len(t_not_s) != 0) and (i+1!=len(t_not_s)):
            if t_not_s[i]+1 == t_not_s[i+1]:
                i+=1
            else:
                sub1.append(t_not_s[:i+1])
                t_not_s = t_not_s[i+1:]
                i=0
        if t_not_s != []: sub1.append(t_not_s)

        i = 0
        while (len(s_not_t) != 0) and (i+1!=len(s_not_t)):
            if s_not_t[i] + 1 == s_not_t[i + 1]:
                i += 1
            else:
                sub2.append(s_not_t[:i+1])
                s_not_t = s_not_t[i+1:]
                i = 0
        if s_not_t != []: sub2.append(s_not_t)

        # 종성 혹은 부호만 따로 추가로 추출되는 경우 뒤에서 후처리 해줌
        char = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ',
                'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        char += [".", ",", "?", "!"]
        # for s1 in sub1:
        #     if (len(s1)==1) and (text[s1[0]] in char):
        #         sub1.remove(s1)
        print(sub1)
        for s2 in sub2:
            if (len(s2)==1) and (sentence[s2[0]] in char):
                sub2.remove(s2)
        print(sub2)

        assert len(sub1) == len(sub2)
        # if len(sub1) != len(sub2):
        #     sub = [[],[]];
        #     new = [[i for i, x in enumerate(text) if x in sentence], [i for i, x in enumerate(sentence) if x in text]]
        #     for ni,n in enumerate(new):
        #         i = 0
        #         while (i + 1 != len(n)):
        #             if n[i] + 1 == n[i + 1]:
        #                 i += 1
        #             else:
        #                 sub[ni].append(n[:i + 1])
        #                 n = n[i + 1:]
        #                 i = 0
        #         if n != []: sub1.append(n)
        #         print(sub[ni])
        #

        for i in range(0,len(sub1)):
            outputs.append([sub1[i], sub2[i]])

        #sen2text: {sentence의 어절 위치: [그에 해당하는 text의 위치들]}
        sen2text = {i:[] for i,sen in enumerate(sentence)};

        # outputs에 포함되지 않은 text의 중복단어 인덱스
        t_more_than = {}
        for i, x in enumerate(text):
            if ((text.count(x) >= 2) and ((i not in re_t_not_s) and (i not in re_t_not_s))):
                if x not in t_more_than.keys(): t_more_than[x] = [i]
                else: t_more_than[x].append(i)
        #print(t_more_than)

        # outputs에 포함되지 않은 text와 sentence에 대하여
        # 중복 어절일 경우 같은 text지만 위치가 다른 경우가 존재
        # 중복 어절의 위치를 정확하게 맞춰주기 위한 부분
        for i in [i for i in range(0, len(sentence)) if i not in re_s_not_t]:
            for t in [j for j in range(0, len(text)) if j not in re_t_not_s]:
                if sentence[i] == text[t]:
                    if (text.count(text[t]) > 1):
                        if text[t] in t_more_than: sen2text[i] = [t_more_than[text[t]][0]]
                        del t_more_than[text[t]][0]
                    else: sen2text[i] = [t]
                    if (sen2text[i] != []):break
        #print(sen2text)

        # outputs에 맟춰 sentence와 text 어절의 위치 정보
        for output in outputs:
            s = 0;
            t = 0;
            if len(output[0]) == 1:
                for out in output[1]: sen2text[out] = output[0]
            else:
                len_t = len(text[output[0][t]])
                len_s = len(sentence[output[1][s]])
                while ((t != len(output[0])) and (s != len(output[1]))):
                    #print("t: "+str(t)+", s: "+str(s))
                    #print(len_t)
                    #print(len_s)
                    if (len_t == len_s):
                        sen2text[output[1][s]].append(output[0][t])
                        t+=1;s+=1;
                        if (s != len(output[1])) and (t != len(output[0])):
                            len_t = len(text[output[0][t]])
                            len_s = len(sentence[output[1][s]])
                            continue
                    elif (len_t < len_s):
                        sen2text[output[1][s]].append(output[0][t])
                        len_s -= len_t
                        t += 1
                        if (len_s <= 0):
                            s += 1
                            if s != len(output[1]): len_s = len(sentence[output[1][s]])
                            continue
                        if t != len(output[0]):len_t = len(text[output[0][t]])
                    else:
                        sen2text[output[1][s]].append(output[0][t])
                        len_t -= len_s
                        s += 1
                        if (len_t <= 0):
                            t += 1
                            if t != len(output[0]): len_t = len(text[output[0][t]])
                            continue
                        if s != len(output[1]): len_s = len(sentence[output[1][s]])

                    #print(sen2text)

    # 후처리
    #빈 리스트로 된 sen2text는 앞의 인덱스 키의 가장 마지막 인덱스의 값과 동일하게 받는다.
    for s,t in sen2text.items():
        if t == []:
            sen2text[s] = [max(sen2text[int(s)-1])]

    print(sen2text)
    return idx2text_word, text_word2idx, idx2sen_word, sen_word2idx, sen2text


if __name__ =='__main__':
    #with open("./../../data/klue-nli-v1_dev.json", "r", encoding="utf-8") as inf:
    with open("./../../data/klue-nli-v1_train.json", "r", encoding="utf-8") as inf:
    #with open("./data/postpocess_prem_other.json", "r", encoding="utf-8") as inf:
        datas = json.load(inf)

    final_result = []; all_koala = [];
    for id, data in tqdm(enumerate(datas)):
        outputs = []
        koala = []
        texts = [data["premise"], data["hypothesis"]]
        #texts = [data["premise"]]+data["other"]["hypothesis"]
        for text in texts:
            text = text.replace(" 에서 ", "에서 ")
            print(text)

            # 의존구문분석 활용
            # sentence: 의존 구문분석 후 분리된 origin text의 어절
            # result = {"root": verb, "words":의존구분분석 결과 리스트}
            sentences, results = sentences_dependency_parser('hnn', text)
            print(results)

            sentence = " ".join([str(sentence) for sentence in sentences])
            sentence = sentence.replace("\xa0"," ")

            new_results ={"root":[result["root"] for result in results], "words": sum([result["words"] for result in results],[])}
            result = new_results

            for k, word in enumerate(result["words"]):
                result["words"][k][0][0] = result["words"][k][0][0].replace("\xa0", " ")
                result["words"][k][0][1] = result["words"][k][0][1].replace("\xa0", " ")

            koala_dic = {"sentence": str(sentence), "result": str(result)}
            print("origin text: "+text)
            print(result)

            text = text.split()
            re_text = text.copy()
            print('text')
            print(text)

            output = []
            #outputs.append(result["root"])
            sentence = sentence.split()
            print('sentence')
            print(sentence)

            #print([r for r in result["words"]])
            # origin text와 sentence 사이의 정보를 원활히 바꿀 수 있도록 사용
            idx2text_word, text_word2idx, idx2sen_word, sen_word2idx, idx_sen2text = sen2text(text, sentence)
            print("idx_sen2text: "+str(idx_sen2text))

            for i,word in enumerate(result["words"]):
                print(word[0])
                word1_idx = [j for j, w in enumerate(sentence) if w == word[0][0]]
                word2_idx = [j for j, w in enumerate(sentence) if w == word[0][1]]
                result["words"][i].append([[], []])

                print(word1_idx)
                print(word2_idx)

                min_len = len(sentence)
                for idx1 in word1_idx:
                    for idx2 in word2_idx:
                        if min_len > abs(idx1-idx2):
                            min_len = abs(idx1-idx2)
                            result["words"][i][0][0] = " ".join([text[j] for j in idx_sen2text[idx1]])
                            result["words"][i][0][1] = " ".join([text[j] for j in idx_sen2text[idx2]])
                            result["words"][i][-1][0] = sorted(idx_sen2text[idx1])
                            result["words"][i][-1][1] = sorted(idx_sen2text[idx2])

            print(result)
            koala.append(result)
        all_koala.append(koala)

        # if (0<id) and (id % 1000 == 0):
        #     sub_result = []
        #     if id == 1000: new_datas = datas[id-1000:id+1]
        #     else: new_datas = datas[id-1000+1:id+1]
        #     for i,(data, koala) in enumerate(zip(new_datas, all_koala)):
        #         data["premise"] = {"origin": data["premise"], "koala": koala[0]}
        #         data["hypothesis"] = {"origin": data["hypothesis"], "koala": koala[1]}
        #         sub_result.append(data)
        #     with open("result1/train/koala_1_klue_nli_train_sub_{}.json".format(str(int(data["guid"].split("klue-nli-v1_train_")[1]))), 'w', encoding="utf-8") as f:
        #         json.dump(sub_result, f, ensure_ascii=False, indent=4)
        #     print("add the file from {}th data to {}th data".format(str(id-1000+1),str(id)))



    for i,(data, koala) in enumerate(zip(datas, all_koala)):
        datas[i]["premise"] = {"origin":data["premise"], "koala":koala[0]}
        datas[i]["hypothesis"] = {"origin": data["hypothesis"], "koala":koala[1]}

    #with open("./../../data/koala/koala_ver1_klue_nli_dev.json", 'w', encoding="utf-8") as f:
    with open("./../../data/koala/koala_1_klue_nli_train_last.json", 'w', encoding="utf-8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)


