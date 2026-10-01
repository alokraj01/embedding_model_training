I want to create embedding training approach. 
Fine-tune bge-base-en-v1.5 or bge-small with sentence-transformers. It runs fine on MPS, no MLX needed.

Step 1: Build training pairs from Spider or BIRD. Parse the gold SQL to see which tables it uses. Each pair is (question, table description with column names).
Hard negatives are the other tables from the same database. This is where most of the gain comes from, so don't skip it.
Use MultipleNegativesRankingLoss. Wrap it in MatryoshkaLoss if you want smaller vector sizes to still work, which is a nice extra to show.
Compare against base bge and maybe one API embedding model. Measure recall@3 and recall@5 on held-out databases the model never saw.