# ee547-hw2-linlinhua

The README.md in your repository root must contain:
1. LinLin Hua. linlinhu@usc.edu
2. problem3: AWS_PROFILE=aws_inspector aws sts get-caller-identity
3. Brief description of your embedding architecture (Problem 2)
   
I take each paper’s abstract, lowercase it, tokenize with a lightweight regex, and build a vocabulary. The model is a word2vec-style skip-gram: given a center word, predict nearby words within a small window (about 2). I add a handful of negative samples per positive (around 5) so the model learns to separate unrelated words. The network is just two embedding tables (input/output), typically 100-dim, scored by a dot product and passed through a sigmoid. Training uses binary cross-entropy with Adam, mini-batches of 32, for about 50 epochs (all configurable). After training, I represent each paper by mean-pooling the word vectors from its abstract (skip OOV tokens; if empty, fall back to zeros). I save three artifacts: the word embedding matrix, the vocab (word↔id), and one embedding per paper. It’s a no-frills approach, but it reliably captures distributional semantics and gives a solid baseline paper embedding with very little complexity.
