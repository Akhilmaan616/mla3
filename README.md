Q1:::: 

1. Nature of the data
Aspect	Linux Kernel (C++/C code)	Shakespeare Text
Structure	Highly structured, rule-based, with formal grammar (syntax, types, braces, keywords, etc.)	Semi-structured, natural language with flexible syntax, poetic meter, and ambiguity
Vocabulary	Limited, technical, mostly reserved keywords, identifiers, API calls	Vast, expressive, includes archaic terms, idioms, and free-form expressions
Dependencies	Strong long-range dependencies (variable declarations, scopes, types)	Weaker structural dependencies, but strong semantic and rhythmic dependencies
Error Tolerance	Very low — a wrong token breaks compilation	High — humans tolerate creative or unusual phrasing
2. Model behavior and architecture
Aspect	Code Prediction (Linux Kernel)	Word Prediction (Shakespeare)
Tokenization	Lexical tokens (identifiers, operators, keywords, literals)	Word-level or subword-level tokens
Context Modeling	Needs to track syntactic context, indentation, and scope hierarchy	Needs to track semantic meaning, emotional tone, rhyme, and meter
Training Objective	Next-token prediction under strict syntax and semantics	Next-word prediction under stylistic and semantic continuity
Model examples	OpenAI Codex, GitHub Copilot, Code Llama, DeepSeekCoder	GPT-2, GPT-3 fine-tuned on Shakespeare corpus
Evaluation metrics	AST correctness, compilation success, functional accuracy	Perplexity, BLEU, stylistic similarity
3. Kernel vs. Shakespeare in model performance
Category	Code Models (Linux Kernel)	Language Models (Shakespeare)
Entropy of next token	Low — few valid tokens given context	High — many valid continuations
Predictive accuracy	Can exceed 90% for next-token accuracy in structured code	Around 40–60% top-1 accuracy in natural text
Model attention	Focuses on syntax trees, definitions, type hierarchies	Focuses on word embeddings, semantics, rhythm
Error recovery	Must be exact (compiler-enforced)	Can be stylistically flexible
4. Summary
Feature	        Linux Kernel C++ Prediction    	 Shakespeare Next-Word Prediction
Domain	       Programming / formal language	      Natural / literary language
Grammar	           Deterministic	                      Probabilistic
Valid next tokens  	Very limited	                      Very open
Model goal	        Correctness	                        Creativity
Difficulty	      High syntactic precision	          High semantic ambiguity



Q2::::

Effect of L1 Regularization
1. Sparsity
Adds penalty proportional to |w| (absolute weights).
Encourages many weights to become exactly zero.
Produces sparse models — only key features remain active.
Acts as a form of feature selection.
2.  Boundary jaggedness
Uses fewer active features → less smooth, more piecewise-linear boundaries.
Decision boundary can appear jagged or sharp in regions with complex data.
May fit tightly to training data, capturing local variations.

Effect of L2 Regularization
1. Smoothness
Adds penalty proportional to w² (squared weights).
Shrinks all weights but does not make them zero.
Leads to smooth, gradual decision boundaries.
Reduces overfitting by discouraging large parameter values.
2. Margin
Promotes larger, more stable margins between classes.
Boundary is less sensitive to noise and outliers.
Improves generalization to unseen data.
