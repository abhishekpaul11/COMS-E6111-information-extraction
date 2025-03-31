# COMS-E6111-information-extraction
An Information Extraction system based on the Google Custom Search API which uses SpanBERT and Google Gemini API to extract tuples with the Iterative Set Expansion (ISE) algorithm.

## Team
1. Abhishek Paul (ap4623)
2. Puja Singla (ps3467)

## Files submitted
1. [pytorch_pretrained_bert](/pytorch_pretrained_bert) --> Folder that contains helper methods for the SpanBert model
2. [transcripts](/transcripts) --> Folder that contains all the transcript text files for the 2 given test cases using spanbert and gemini
   1. [spanbert.txt](/transcripts/spanbert.txt) --> Transcript for seed query "bill gates microsoft" over relation "Work_For" with a confidence threshold of 0.7 and k-value of 10 (using the spanbert model)
   2. [gemini.txt](/transcripts/gemini.txt)  --> Transcript for seed query "bill gates microsoft" over relation "Work_For" and k-value of 10 (using the gemini model)
3. [.gitignore](.gitignore) --> gitignore file for python projects
4. [download_finetuned.sh](download_finetuned.sh) --> shell script to install the pretrained SpanBert model
5. [extract_gemini.py](extract_gemini.py) --> helper file to perform information extraction using gemini
6. [extract_spanbert.py](extract_spanbert.py) --> helper file to perform information extraction using spanbert
4. [google_search.py](google_search.py) --> sub-routine for performing the google search
5. [LICENSE](LICENSE) --> MIT License
6. [main.py](main.py) --> the main entry point of the application
7. [README.md](README.md) --> Project Readme file
8. [requirements.txt](requirements.txt) --> list of dependencies and external libraries used
9. [scrape_text.py](scrape_text.py) --> sub-routine for scraping text for information extraction from the urls returned by the google search
10. [spacy_help_functions.py](spacy_help_functions.py) --> helper file to use spaCy for text processing

## Installation and Execution

You can clone the repo using the command given below,

```bash
git clone https://github.com/abhishekpaul11/COMS-E6111-information-extraction.git
```
or download a zip file of this repo.

Once, you're in the repository in your terminal, type the following commands

**Note:** <br>
The project uses the spaCy NLP library which is most compatible with Python 3.12.9. In case you are having installation issues, kindly **switch to
Python 3.12.9**

### Create a virtual environment

```bash
python3 -m venv venv
```

### Activate the virtual environment

```bash
source venv/bin/activate
```

### Install all the required dependencies

```bash
pip3 install -r requirements.txt
```

### Install the pretrained SpanBert model

You might need to install `wget` tool in your system before doing this.

```bash
bash download_finetuned.sh
```

### Install the large spaCy model trained on English for NLP tasks

```bash
python3 -m spacy download en_core_web_lg
```

### Run the application

```bash
python3 main.py [-spanbert|-gemini] <r> <t> <q> <k>
```

where,<br>
`[-spanbert|-gemini]` is one of "-spanbert" or "-gemini" indicating the method you'd like to use for Information Extraction, <br>
`<r>` is an integer between 1 and 4, indicating the relation to extract: 1 is for Schools_Attended, 2 is for Work_For,3 is for Live_In, and 4 is for Top_Member_Employees,<br>
`<t>` is a real number between 0 and 1, indicating the "extraction confidence threshold", which is the minimum extraction confidence that we request for the tuples in the output; t is ignored if we are specifying -gemini, <br>
`<q>` is a "seed query", which is a list of words in double quotes corresponding to a plausible tuple for the relation to extract (e.g., "bill gates microsoft" for relation Work_For), <br>
`<k>` is an integer greater than 0, indicating the number of tuples that we request in the output.

**Note**<br>
1. The above instructions are mentioned for macOS / Linux systems. Run the appropriate commands if you are using a Windows System.
2. You need to add your API Keys for Google Custom Search, Google Programmable Search Engine and Google Gemini in [main.py](/main.py) for the project to be functional.

**Assumption**<br>
You have python3 and pip3 available in your system.
If not, you can get it from [Python Downloads](https://www.python.org/downloads/) and
[Pip Installation](https://pip.pypa.io/en/stable/installation/) respectively.


## Project Design

### main.py

This is the entry point of the application.
1. It parses the CLI arguments and performs the Google Search by calling [this](#google_searchpy) subroutine.
2. After filtering out the non-html results, it sends the url from each result to [scrape_text.py](#scrape_textpy).
3. It returns the pre-processed and cleaned text extracted from the url.
4. This extracted text chunk is sent to [spaCy](#has_entities_of_interest) to filter out the sentences that have the entities of our interest based on the relation the user has 
provided in the CLI prompt. (For example, PERSON and ORGANISATION in case of Work_For).
5. If the chosen method is 'spanbert', these valid sentences are then sent to [extract_spanbert.py](#extract_spanbertpy) to get the tuples with the required relation and confidence level above the threshold, the details of which can be
found in the [next section](#using-spanbert-model).
6. If the chosen method is 'gemini', these valid sentences are then sent to [extract_gemini.py](#extract_geminipy) to get the tuples with the required relation, the details of which can be
found in the [next section](#using-gemini-model).
7. These new tuples are added to a set after removing the duplicates and a new query is chosen to repeat the process. The specifics can be found in the [next_section](#information-extraction-method).

It terminates the application under these scenarios:
1. Stalling - If no unused queries are left for iterative set expansion process to continue.
2. 'k' unique tuples (with confidence level above the threshold in case of spanbert) have been found.

### google_search.py

This simply uses the Google Custom Search JSON Api Key and the Programmable Search Engine ID to perform the google search on the given query
and returns the results with appropriate error handling.

### scrape_text.py

Given a URL, it extracts the text content from it, cleans it up (removes unnecessary whitespaces, newline characters and non-printable characters) and returns the first 10,000 characters from it.

### extract_gemini.py

Helper method that makes the Gemini API call to the Gemini 2.0 Flash Model for extracting tuples from a sentence, given a relation.

### extract_spanbert.py

Helper method that returns tuples along with its confidence level from a sentence for a given relation by invoking the SpanBert model.

### spacy_help_functions.py

Helper file with sub-routines for performing IE tasks on text.

#### get_entities()

Returns all named entities in a sentence after named entity tagging.

#### has_entities_of_interest()

Returns true if the passed sentence contains all the entities based on the relation the user has provided in the CLI prompt. For example, for Work_For, it returns true if the sentence contains a PERSON and an ORGANISATION entity.

#### extract_relations()

Returns a list of all tuples from the provided text for the given relation whose confidence level is above the threshold.

#### create_entity_pairs()

It accepts a sentence and returns all possible combinations of pairs of entities of our interest (as explained above) present in the sentence.

### spanbert.py

Code related to the SpanBert Model

## Information Extraction Method

1. The text content is extracted from each url (which has not been processed before) returned after the google search performed on the seed query given the by the user in the CLI prompt for the initial iteration (or the generated query for subsequent iterations).
2. The text content is cleaned by removing unnecessary whitespaces, newline characters and non-printable characters. It is then trimmed to the first 10,000 characters, which will be used for IE going forward.
3. Using spaCy, the text chunk is broken into individual sentences. For each sentence, spaCy returns the list of named entities it contains. We are only
interested in those sentences which at least has all the entities involving the relation specified by the user in the CLI prompt. We filter out these valid
sentences and carry on our process with them.

### Using SpanBert Model

1. If the user has specified spanbert as the method for information extraction, we try to generate all possible combinations of pairs of entities for the relation in the valid sentence.
2. We then filter out only those pairs which have an exact match with the relation type. For example, for Work_For the subject should be a PERSON and object should be an ORGANISATION, all (PERSON, ORGANISATION) entity pairs are kept while removing
all other pairs like (PERSON, PERSON), (ORGANISATION, PERSON) and (ORGANISATION, ORGANISATION).
3. The filtered entity pairs are then sent to the SpanBert model along with the parent sentence.
4. It returns the relation between the pairs and the confidence level.
5. We then pick only those relations which match with the user specified relation and have a confidence level above the threshold. If duplicates are found, we keep the one with the higher confidence level.
6. After repeating the process for every valid sentence in every url, we get a bunch of relevant tuples. Again, if duplicates are found, we keep the one with the higher confidence level.
7. The program terminates if the number of accumulated tuples from the above step has reached `<k>` as specified by the user.
8. If not, we pick the tuple with the highest confidence level (such that this tuple has not been used as a prior query) as the query for the next iteration and proceed to repeat the steps from the top.
9. In case we are left with no tuples which have not been used as a query before, it means the process has stalled. The program terminates in that case returning the tuples that have been gathered so far.

### Using Gemini Model

1. If the user has specified gemini as the method for information extraction, we use the Gemini API to invoke the Gemini 2.0 Flash model.
2. We have fabricated a one-shot in-context learning prompt to extract a list of all tuples from a sentence for a specific relation, which can be found at [extract_gemini.py](/extract_gemini.py). We will be using this prompt to perform the IE task.
3. We pass every valid sentence in an url to Gemini, and it returns to us a list of (subject, object) tuples that satisfy the given relation in the sentence. We remove the duplicate tuples, keeping only one copy of them.
4. After repeating the process for every url, we get a bunch of relevant tuples. Again, if duplicates are found, we keep only one copy of them, removing the rest.
5. The program terminates if the number of accumulated tuples from the above step has reached `<k>` as specified by the user.
6. If not, we arbitrarily pick a tuple (such that this tuple has not been used as a prior query) as the query for the next iteration and proceed to repeat the steps from the top.
7. In case we are left with no tuples which have not been used as a query before, it means the process has stalled. The program terminates in that case returning the tuples that have been gathered so far.

## External Libraries Used

### [requests](https://pypi.org/project/requests/)

To perform the Search Engine API call and get the results.

### [beautifulsoup4](https://pypi.org/project/beautifulsoup4/)

To scrape text from the urls returned by the Google search.

### [spaCy](https://spacy.io/)

To perform NLP Pre-processing tasks (tokenization and Named Entity Tagging) on the plain text before Information Extraction. 

### [Pretrained SpanBert Model](https://github.com/facebookresearch/SpanBERT)

To predict the relation between two entities in a sentence along with its confidence level.

### [google-generativeai](https://pypi.org/project/google-generativeai/)

To extract tuples for a given relation from a sentence.

## Additional Information

### Handling of non-html files

We have decided to **ignore the non-html files** in the information extraction analysis. 