Augmented Max-margin Markov Network Toolkit, with extended features and function support, and some simple performance optimization (about 3~4 times as fast as the original toolkit):

New feature templates:
(Note: all comparison templates accept any number of input x, the output is the equality test result of every adjacent pair of x. For example, if the input is "is is was was" and the template is "%==%x[-3,0]%x[-2,0]%x[-1,0]%x[0,0]", the output will be the concatenation of ==, != and ==. i.e. "==!===")
  1. %== : for testing whether the input elements are the same
  1. %~= : for testing whether the input elements are the same, if they are the same, include the token identity, i.e. both the current POS and the previous POS are VERB
  1. %!= : for testing whether the input elements are different
  1. %x[-2,1,5!=FL] : the 2nd previous Column\_1 feature with Column\_5 not being FL (all indices start from 0), it will skip intermediate nodes with Column\_5 being FL
  1. %x[-2,1,5==FL] : the 2nd previous Column\_1 feature with Column\_5 being FL (all indices start from 0), it will skip intermediate nodes with Column\_5 not being FL
  1. %y[-2,1] : the 2nd previous node's label on the second layer of y, (FM3N only), the 2nd parameter is the y's layer ID: default=0, 1st layer; 1 means the 2nd layer.


Templates can now have attributes (append the following keyword(s) on the same line):
  1. no\_header : do not put template header so that features from different templates can match. It can extract this kind of feature: the word 'uh' has occurred within 5 words from the current position.
  1. sort\_x : sort observations. It can extract this kind of feature: the previous token and the current token are 'was' and 'is', or 'is' and 'was'
  1. freq\_thresh=### : override the default freq\_thresh for features under this template, i.e., template-specific frequency pruning


New functions:
  1. support computing scores for given hypotheses
  1. support output n-best list and label posteriors
  1. support dumping the model parameters in a more informative way
  1. automatically compress/decompress files ending with .gz for I/O
  1. support using - for standard input and output (not for all)
  1. max-marginal KL divergence feature selection (not very effective so far)
  1. support multiple layers of labels, (Factorial Max-margin Markov Networks (FM3N) using Viterbi Inference)

Wang Xuancong (National University of Singapore)