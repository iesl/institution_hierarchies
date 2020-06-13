import codecs
import sys
import string
from main.objects.Tokenizer import Char
from main.objects.Tokenizer import Unigram
from main.objects.Tokenizer import UnigramUC


def count_tokens(read_file, write_file, tokenizer_name, min_count):
    '''
    Counts the number of each token in the file and then writes the tokens that occur more than a threshold to form the vocab
    param read_file: file to read tokens 
    param write_file: file to write vocab
    param tokenizer_name: tokenizer to use 
    param min_count: threshold of minimum number of occurences of tokens for vocab
    '''
    tokenizer = None
    if(tokenizer_name == "Char"):
        print("Made Char")
        tokenizer = Char()
    elif(tokenizer_name == "Unigram"):
        tokenizer = Unigram()
    elif(tokenizer_name == "UnigramUC"):
        tokenizer = UnigramUC()


    token_dict = {}

    counter = 0

    with open(read_file, 'r+') as rf:
        for line in rf:
            splt = line.strip().split("\t")
            if counter % 1000 == 0:
                sys.stdout.write("\rProcessed {} lines".format(counter))
            for s in splt:
                s_tokens = tokenizer.tokenize(s)
                for token in s_tokens:
                    if token not in token_dict:
                        token_dict[token] = 1
                    else:
                        token_dict[token] += 1

                counter += 1
    
    sys.stdout.write("\nDone....Now Writing Vocab.")
    
    with codecs.open(outputfile, "w+", "UTF-8") as wf:
        
        token_id = 2
        for token in token_dict.keys():
            if token_dict[token] >= min_count:
                wf.write("{}\t{}\n".format(token, token_id))
                token_id += 1

        wf.flush()
        wf.close()


if __name__ == "__main__":
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    tokenizer_name = sys.argv[3]
    min_count = int(sys.argv[4])

    outputfile = outputfile + "_" + tokenizer_name.lower()
    count_tokens(inputfile, outputfile, tokenizer_name, min_count)