import os
this_dir = os.path.dirname(__file__) 

class English_Numbers_Symbols_Specials_extended: 
    def __init__(self):
        print("English_Numbers_Symbols_Specials_extended")
        numbers = list(range(48,58))
        alphabets = list(range(65,91)) + list(range(97,123))
        symbols_str = ["(",")","[","]","#","/","_","-",">","<","\\","\"",",","*","'","?","^","@","$","!","%","&","=","+","|",";",":",".", "~", "{", "}"]
        specials_str = ["⑦","ⓓ","♀","★","ⓔ","°","■","℉","◇","「","⑪","↕","ⓚ","ⓡ","ⓐ","⑬","⑫","◐","ⓘ","㉿","ⓞ","ⓣ","▽","⑧","②","※","♠","≡","¶","ⓤ","◑","』","±","④","≠","⑮","▼","↔","ⓢ","▷","⊙","③","ⓜ","ⓧ","♣","♪","↓","○","∞","●","ⓑ","⑩","￥","ⓗ","♤","≒","☎","㈜","¢","￡","®","©","▣","□","☞","ⓟ","◎","ⓨ","ⓦ","⑥","◁","ⓛ","】","⑨","⑭","≤","ⓠ","♩","ⓒ","ⓙ","ⓕ","←","♡","℃","↑","◀","⑤","·","☜","♥","▶","」","≥","◈","『","÷","◆","ⓥ","∴","ⓖ","ⓩ","☏","♬","€","㉾","☆","♨","△","ⓝ","【","♧","→","▲","①","♂"]
        other_str = [chr(n) for n in numbers+alphabets]
        other_str.extend(symbols_str)
        other_str.extend(specials_str)
        symbol_asc = [ord(x) for x in symbols_str]
        specials_asc = [ord(x) for x in specials_str]
        key = numbers+alphabets+symbol_asc+specials_asc
        key_str = [str(k) for k in key ]
        val = range(len(key))
        val_str = [str(v) for v in val]
        self.pair = dict(zip(key_str,val))
        self.inv_pair = dict(zip(val_str,key))
        self.num_classes = len(val) + 1
        self.key = key
    def encoder(self,label):
        final_label = []
        for c in label:
            index = self.pair[str(ord(c))]
            final_label.append(index)
        return final_label
    def decoder(self,label):
        decoded_label = [j for j in label if j != -1]
        str_pred = ''.join([chr(self.inv_pair[str(l)]) for l in decoded_label])
        return str_pred
    def is_valid(self,label):
        if not all([ord(ch) in self.key for ch in label]):
            return False
        else:
            return True
