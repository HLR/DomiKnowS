Qiang Ning's Original:

This directory TBAQ-cleaned contains cleaned and improved AQUAINT and TimeBank corpus. We updated these corpora with the following changes. 

Common changes: i. Cleaned formatting for all files. All the files are in same format. Easy to review/read, ii. Made all files XML and TimeML schema compatible, iii. Some missing events and temporal expressions are added. 

AQUAINT changes: i. Added event-DCT temporal relations

TimeBank changes: i. Events are borrowed from the TempEval-2 corpus, ii. Temporal relations are borrowed from actual TimeBank corpus, which contains a full set of TimeML temporal relations. iii. Along with our correction, also added temporal expressions correction suggestion from Kolomiyets et al. (2011) (total additional 10 temporal expressions from them).

Added by why16gzl:

The data is actually from TempEval3 https://www.cs.york.ac.uk/semeval-2013/task1/index.php%3Fid=data.html
We did not find the annotated file named "nyt_20130321_sarcozy.tml", but only find the original text. Hence we temporarily eliminate this file from MATRES, and now training set (timebank) has 183 documents, validation set (aquaint) has 72 documents, and test set (platinum) has 19 documents (20 in the original).

The preprocessing for MATRES begins from line 275 in document_reader.py

