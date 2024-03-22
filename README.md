# OPJ-2023-24
## Week 1: 1
## Week 2: 8
## Week 3: 15.03.24: Corpus collection 
## Week 4: 22.03.24: Corpus collection complete 
## Week 5: 29.03.24: Data cleaning complete
## Week 6: 05.03.24: Initial annotation campaign with 150 instances done.
## Week 7: 12
## Week 8: 19
## Week 9: 26
## Week 10: 03
## Week 11: 
## Week 12:


# Project Steps
1. Scraping/Data collection. **Data Size**: Start with a manageable amount (e.g., 1000-5000 sentences) and increase if required. Either use:
  - Web-scraping with selenium  or
  - Manual scraping
2. Data organisation 
  - file formats 
3. Task description - [informal]
  - Goal: Are you building a sentiment analysis model for product reviews, social media analysis, or something else?
  - Target Sentiment: Are you interested in just positive/negative sentiment, or a wider range (e.g., very positive, neutral, very negative)?
4. Data cleaning - Pre
  - Collectively decide and **document decisions** taken for creating dataset. 
  - Follow the instructions. These are the same as those detailed in the presentation.   
    - [Sinclair, J. (2004). Corpus and Text – Basic Principles. U: Wynne, M.,  DevelopingLinguistic Corpora: A Guide to Good Practice.](http://users.ox.ac.uk/~martinw/dlc/)
    - [Developing Linguistic Corpora: a Guide to Good Practice](http://icar.cnrs.fr/ecole_thematique/contaci/documents/Baude/wynne.pdf) . 
  - Removal of metadata (yes/no)
  - How Data looks 
  	- Sentence structure.
  	- Put each sentence on one line
  	- Do you wish to handle conditionals, speculations.
- Annotation guidelines
  - prepare the guidelines with scheme. Use the example put in omega or [Example 1 for negation detection](https://github.com/ltgoslo/norec_neg/blob/main/annotation_guidelines/guidelines_neg.md)
  - Take 150 random instances
  	- All group members perform independent annotation. 
  	- sit together and discuss abnormalities, problems.
    - note down the problems and solutions decided.   
  	- Update the annotation guidelines
- ### TBD by 05-04-24
-----------------------------------------------------------------
- Re-annotate the whole dataset (3k-5k)
7. Data cleaning- Post [if required]
8. Exploratory data analysis 
  - Data distribution 
  - Size – Tokens and sentences/documents 
9. Problem definition - [formal]
  - Inter-annotator Agreement
  - Label distribution 
10. Modelling/Training 
11. Performance metric 
12. Error Analysis 
13. Data Serving/Demo 

- **Some steps might involve multiple iterations like preprocessing/cleaning**
- Incase you finish one step you can read about next step.

## Instructions
- upload all the files on github.
- keep track of all the activities and time (in hrs spent by each member each week).
- document all steps.

## Final Delieverables
- Code/Demo
- REPORT
- PRESENTATION (8-10 min)


**Useful Links**
- [Learn Markdown/Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

  
# Tools
- [Classla](https://pypi.org/project/classla/)
- [Stanza](https://stanfordnlp.github.io/stanza/neural_pipeline.html)
- [Spacy](https://spacy.io/models/hr)
  
# Groups
- https://github.com/laracoen/MKLPM/blob/main/README.md
- https://github.com/BoViNiMa/OPJ/tree/main 
- https://github.com/Sentimentalci/opjprojekt
- https://github.com/My-Croatian-is-better-than-yours/SA-Comments/tree/main
- https://github.com/Serious-bus1ness/Uvod-u-projekt  
