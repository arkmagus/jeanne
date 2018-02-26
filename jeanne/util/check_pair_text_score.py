import sys

if __name__ == '__main__':
    """
    usage : python check_pair_text_score,py text_file score_file
    """
    file_a = sys.argv[1]
    file_b = sys.argv[2]

    lines_a = open(file_a).readlines()
    lines_b = open(file_b).readlines()

    assert len(lines_a) == len(lines_b), ('total lines is different !')

    for ii in range(len(lines_a)) :
        assert len(lines_a[ii].strip().split()) == len(lines_b[ii].strip().split()), \
        ("number of words in lines {} is different {} =/= {} \nsent A : {} sent B : {}".format(ii, len(lines_a[ii].strip().split()), len(lines_b[ii].strip().split()), lines_a[ii], lines_b[ii]))
    print('===FINISH===') 
