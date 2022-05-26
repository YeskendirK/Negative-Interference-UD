import subprocess

expmix_langs = ["UD_Armenian-ArmTDP",
         "UD_Breton-KEB",
         "UD_Buryat-BDT",
         "UD_Faroese-OFT",
         "UD_Kazakh-KTB",
         "UD_Upper_Sorbian-UFAL",
         "UD_Arabic-PADT",
         "UD_Czech-PDT",
         "UD_English-EWT",
         "UD_Finnish-TDT",
         "UD_French-Spoken",
         "UD_German-GSD",
         "UD_Hindi-HDTB",
         "UD_Hungarian-Szeged",
         "UD_Italian-ISDT",
         "UD_Japanese-GSD",
         "UD_Korean-Kaist",
         "UD_Norwegian-Nynorsk",
         "UD_Persian-Seraji",
         "UD_Russian-SynTagRus",
         "UD_Swedish-PUD",
         "UD_Tamil-TTB",
         "UD_Urdu-UDTB",
         "UD_Vietnamese-VTB",
         "UD_Bulgarian-BTB",
         "UD_Telugu-MTG"]


for lang in expmix_langs:
    source = "data/ud-treebanks-v2.3/"+lang
    dest = "data/exp-mix/"+lang
    subprocess.call(["cp", "-R", source, dest])
