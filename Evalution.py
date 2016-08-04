from decoder import *

def wer(r, h):
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


def main():
	TIMIT_pkl_file = os.path.join(os.getcwd(),'TIMIT_data_prepared_for_CTC.pkl')
	#TIMIT_pkl_file = os.path.join(os.getcwd(),'TIMIT_data_prepared_for_Test.pkl')

	with open(TIMIT_pkl_file,'rb') as f:
			data = pickle.load(f)
			list_of_alphabets = data['chars']
	tgt = data['y_indices']
	RNN_in, BiRNN = getTrainedRNN()

	CLM_in, CLM_mask, CLM = getTrainedCLM()
	print "Model loaded"
	total_words = 0
	Clm_errors = 0
	argmax_errors = 0
	for i in range(len(data['x'])):

		input_data = data['x'][i];
		pred = BiRNN.eval({RNN_in.input_var: [input_data]})
		clm_decoded = decode(pred[0])
		argmax_decoded = index2char_TIMIT(np.argmax(pred, axis = 2)[0])
		print "setence no. ", i
		print "clm_decoded : " , clm_decoded
		print "argmax_decoded : ",argmax_decoded 
		curr_tgt = np.asarray(tgt[i],dtype=np.int16)

		curr_tgt = index2char_TIMIT(curr_tgt)
		print "Target : ", curr_tgt
		total_words = total_words + len(curr_tgt.split())
		Clm_errors  = Clm_errors + wer(curr_tgt,clm_decoded)
		argmax_errors = argmax_errors + wer(curr_tgt,argmax_decoded)
		print "CLM word_error_rate :" ,float(Clm_errors)/total_words
		print "Argmax word_error_rate :",float(argmax_errors)/total_words
		print "total_words = ",total_words	
if __name__ =="__main__":
	main()
