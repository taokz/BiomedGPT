import csv

with open(".../../../datasets/pretraining/text_pubmed.tsv", 'w') as out_file:
    with open(".../../../datasets/pretraining/pubmed_uncased_sentence_nltk.txt/pubmed_sentence_nltk", 'r') as in_file:
        lines = csv.reader(in_file, delimiter="\n")#, quotechar='"')
        count = 1
        for line in lines:
            out_file.write(str(count) + '\t' + line[0] + '\n')
            if count % 10000 == 0:
                print("finish instance", count)
            count += 1