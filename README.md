# ee547-hw2-linlinhua

The README.md in your repository root must contain:
1. LinLin Hua. linlinhu@usc.edu
2. problem3:
   - AWS_PROFILE=aws_inspector aws sts get-caller-identity
   - My account and region (us-east-1) doesn't has S3 bucket and EC2
     - [create S3 bucket]
      - AWS_PROFILE=aws_inspector aws s3api create-bucket 
      - bucket <your-unique-bucket-name> --region us-east-1 
      - create-bucket-configuration LocationConstraint=us-east-1

   - [EC2]
     - AWS_PROFILE=aws_inspector aws ec2 create-security-group 
     - group-name hw3-sg --description "temp sg" --region us-east-1 
     - query GroupId --output text


   - run your script:
      - AWS_PROFILE=aws_inspector python aws_inspector.py --region us-east-1 --format table

# 3) terminate and clean up
AWS_PROFILE=aws_inspector aws ec2 terminate-instances --instance-ids <i-...> --region us-east-1
AWS_PROFILE=aws_inspector aws ec2 delete-security-group --group-name hw3-sg --region us-east-1

   
   
4. Brief description of your embedding architecture (Problem 2)
   
I take each paper’s abstract, lowercase it, tokenize with a lightweight regex, and build a vocabulary. The model is a word2vec-style skip-gram: given a center word, predict nearby words within a small window (about 2). I add a handful of negative samples per positive (around 5) so the model learns to separate unrelated words. The network is just two embedding tables (input/output), typically 100-dim, scored by a dot product and passed through a sigmoid. Training uses binary cross-entropy with Adam, mini-batches of 32, for about 50 epochs (all configurable). After training, I represent each paper by mean-pooling the word vectors from its abstract (skip OOV tokens; if empty, fall back to zeros). I save three artifacts: the word embedding matrix, the vocab (word↔id), and one embedding per paper. It’s a no-frills approach, but it reliably captures distributional semantics and gives a solid baseline paper embedding with very little complexity.
