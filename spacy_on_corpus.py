# import spacy for nlp
import spacy
# import glob in case user enters a file pattern
import glob
# import shutil in case user enters a compressed archive (.zip, .tar, .tgz etc.); this is more general than zipfile
import shutil
# import plotly for making graphs
import plotly.express as px
# import wordcloud for making wordclouds
import wordcloud
# import json
import json 
# import re
import re
import pyate
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from bertopic import BERTopic
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer



class counter(dict):
    def __init__(self, list_of_items, top_k=-1):
        """Makes a counter.

        :param list_of_items: the items to count
        :type list_of_items: list
        :param top_k: the number you want to keep
        :type top_k: int
        :returns: a counter
        :rtype: counter
        """
        super().__init__()
        #  FROM PROJECT 3c
        # Add each item in list_of_items to this counter (2 lines)
        for item in list_of_items:
        # HINT: Use the add_item method
            self.add_item(item)
        # Reduce to top k if top_k is greater than 0 (2 lines)
        if top_k > 0:
        # HINT: Use the reduce_to_top_k method
            self.reduce_to_top_k(top_k=top_k)
        # you don't have to return explicitly, since this is a constructor
    def add_item(self, item):
        """Adds an item to the counter.

        :param item: thing to add
        :type item: any
        """
        #  FROM PROJECT 3c
        if item not in self:
            self[item]=0
        self[item]+=1
        # HINT: use self[item], since a counter is a dictionary
         # remove pass

        
    def get_counts(self):
        """Gets the counts from this counter.

        :returns: a list of (item, count) pairs
        :type item: list[tuple]
        """
        #  FROM PROJECT 3c
        return list(self.items())

    
    def reduce_to_top_k(self, top_k):
        """Gets the top k most frequent items.

        :param top_k: the number you want to keep
        :type top_k: int
        """
        #  FROM PROJECT 3c
        top_k = min([top_k, len(self)])
        # Sort the frequency table by frequency (least to most frequent) (1 line)
        sorted_keys=sorted(self, key=lambda x: self[x])
        # Drop all but the top k from this counter (2 lines)
        # HINT: go from 0 to len(self)-top_k
        # HINT: use the pop() method; after all, counter is a dictionary!
        for i in range(0, len(self)-top_k):
            self.pop(sorted_keys[i])

class corpus(dict):
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    sentence_sentiment = pipeline("text-classification", model="sst-model", tokenizer=tokenizer)
    classifier=pipeline("sentiment-analysis")
    nlp = spacy.load('en_core_web_md')          
    nlp.add_pipe("combo_basic")     
    summarizer = pipeline("summarization")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
    keybert_model = KeyBERTInspired()
    pos_model = PartOfSpeech("en_core_web_md")
    mmr_model = MaximalMarginalRelevance(diversity=0.3)
    
    def __init__(self, name=''):
        """Creates or extends a corpus.

        :param name: the name of this corpus
        :type name: str
        :returns: a corpus
        :rtype: corpus
        """
        super().__init__()
        #  FROM PROJECT 3c
        # Set or update instance variables (1 line)
        self.name=name
    def get_sentence_level_sentiment(self, doc_id):
        """returns a list of pairs in the format (sentence, label), where label is the sentiment for the sentence
        :rtype: list"""
        sentiment_list=[]
        for sentence in self[doc_id]['doc'].sents:
            sentiment_label = corpus.classifier(sentence.text)[0]['label']
            sentiment_list.append((sentence.text, sentiment_label))
        return sentiment_list
    def render_document_sentiments(self, doc_id):
        """Render a document's sentences and their corresponding sentiment labels as a markdown table.

        :param doc_id: the id of a spaCy doc made from the text in the document
        :type doc_id: str
        :returns: the markdown table
        :rtype: str
        """
        doc = self.get_document(doc_id)
        # Initialize the table with headers
        sentiments_table = "| Sentence | Sentiment |\n| -------- | ---------- |\n"

        # Walk over the sentences in the document
        for sentence, sentiment_label in self.get_sentence_level_sentiment(doc_id):
            # Escape any '|' characters in the sentence to avoid conflicts in markdown table
            sentence_escaped = sentence.replace("|", "\\|")
            # Add the sentence and sentiment label to 'sentiments_table'
            sentiments_table += f"| {sentence_escaped} | {sentiment_label} |\n"

        # Add the row for document-level sentiment
        document_sentiment_label = self[doc_id].get('sentiment', 'N/A')  # Assuming sentiment is stored in the 'sentiment' key
        sentiments_table += f"| Document | {document_sentiment_label} |\n"

        return 'Document Sentiments\n' + sentiments_table

    def get_documents(self):
        """Gets the documents from the corpus.

        :returns: a list of spaCy documents
        :rtype: list
        """
        #  FROM PROJECT 3c
        return [item['doc'] for item in self.values()]
   
    def get_document(self, id):
        """Gets a document from the corpus.

        :param id: the document id to get
        :type id: str
        :returns: a spaCy document
        :rtype: (spaCy) doc
        """
        # FROM PROJECT 3c
        return self[id]['doc'] if id in self.keys() and 'doc' in self[id].keys() else None


                         
    def get_metadatas(self):
        """Gets the metadata for each document from the corpus.

        :returns: a list of metadata dictionaries
        :rtype: list[dict]
        """
        #  FROM PROJECT 3c
        return [doc.get('metadata', {}) for doc in self.values()] # replace None


    def get_metadata(self, id):
        """Gets a metadata from the corpus.

        :param id: the document id to get
        :type id: str
        :returns: a metadata dictionary
        :rtype: dict
        """
        # FROM PROJECT 3c
        return self.get(id, {}).get('metadata', None) # replace None

                         
    def add_document(self, id, text, metadata={}):
        """Adds a document to the corpus.

        :param id: the document id
        :type id: str
        :param doc: the document itself
        :type doc: (spaCy) doc
        :param metadata: the document metadata
        :type metadata: dict
        """
        #  FROM PROJECT 3c
        doc=self.nlp(text)
        self[id]={'doc': doc, 'metadata': metadata, 'sentiment': corpus.classifier(text), 'summary': corpus.summarizer(doc.text, min_length=5, max_length=20)} #'summarizer': corpus.summarizer(text, min_length=)
        # texts=self.get_document_texts()
        # result=classifier(texts)
        # self.update_document_metadata(id,{'sentiment':f"{result}"})


    def get_keyphrase_counts (self, top_k=-1):
        """Builds a keyphrase frequency table
        
        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
        keyphrases=[]
        for doc in self.get_documents():
            keyphrases.extend([keyphrase for keyphrase in list(doc._.combo_basic.keys())])
        return counter(keyphrases, top_k=top_k).get_counts()

    def get_token_counts(self, tags_to_exclude = ['PUNCT', 'SPACE'], top_k=-1):
        """Builds a token frequency table.

        :param tags_to_exclude: (Coarse-grained) part of speech tags to exclude from the results
        :type tags_to_exclude: list[string]
        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
        #  FROM PROJECT 3c
        # Make an empty list of tokens (1 line)
        tokens=[]
        # For each doc in the corpus, add its tokens to the list of tokens (2 lines)
        for doc in self.get_documents():
            tokens.extend([token.text for token in doc if token.pos_ not in tags_to_exclude])
        # Count the tokens using a counter object; return a list of pairs (item, frequency) (1 line)
        # HINT: use the counter class
        return counter(tokens, top_k=-1).get_counts()


    def get_entity_counts(self, tags_to_exclude = ['QUANTITY'], top_k=-1):
        """Builds an entity frequency table.

        :param tags_to_exclude: named entity labels to exclude from the results
        :type tags_to_exclude: list[string]
        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
        #  FROM PROJECT 3c
        # Using get_token_counts as a model, define get_entity_counts using get_documents and a counter object (4 lines of code)
        entities=[]
        # For each doc in the corpus, add its tokens to the list of tokens (2 lines)
        for doc in self.get_documents():
            entities.extend(entity.text for entity in doc.ents if entity.label_ not in tags_to_exclude)
        # Count the tokens using a counter object; return a list of pairs (item, frequency) (1 line)
        # HINT: use the counter class
        return counter(entities, top_k=top_k).get_counts()

    def get_noun_chunk_counts(self, top_k=-1):
        """Builds a noun chunk frequency table.

        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
        #  FROM PROJECT 3c
        # Using get_token_counts as a model, define get_noun_chunk_counts using get_documents and a counter object (4 lines of code)
        noun_chunks=[]
        # For each doc in the corpus, add its tokens to the list of tokens (2 lines)
        for doc in self.get_documents():
            noun_chunks.extend(noun_chunks.text for noun_chunks in doc.noun_chunks)
        # Count the tokens using a counter object; return a list of pairs (item, frequency) (1 line)
        # HINT: use the counter class
        return counter(noun_chunks, top_k=top_k).get_counts()


    def get_metadata_counts(self, key, top_k=-1):
        """Gets frequency data for the values of a particular metadata key.

        :param key: a key in the metadata dictionary
        :type key: str
        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
        #  FROM PROJECT 3c
        metadata_values=[metadata[key] for metadata in self.get_metadatas() if key in metadata]
        # Count the tokens using a counter object; return a list of pairs (item, frequency) (1 line)
        # HINT: use the counter class
        return counter(metadata_values, top_k=top_k).get_counts()
    def get_keyphrase_statistics(self):
        """Prints summary statistics for keyphrases in the corpus.
        
        :returns: the statistics report
        :rtype: str
        """
        text=f'Documents: %i\n' %len(self)
        keyphrase_counts=self.get_keyphrase_counts()
        text+=f'Keyphrases: %i\n' % len(keyphrase_counts)
        return text
    def get_token_statistics(self):
        """Prints summary statistics for tokens in the corpus, including: number of documents; number of sentences; number of tokens; number of unique tokens.
        
        :returns: the statistics report
        :rtype: str
        """
        # NEW FOR PROJECT 4a
        text = f'Documents: %i\n' % len(self)
        text += f'Sentences: %i\n' % sum([len(list(doc.sents)) for doc in self.get_documents()])
        token_counts = self.get_token_counts()
        text += f'Tokens: %i\n' % sum([x[1] for x in token_counts])
        text += f"Unique tokens: %i\n" % len(token_counts)
        return text

    def get_entity_statistics(self):
        """Prints summary statistics for entities in the corpus. Model on get_token_statistics.
        
        :returns: the statistics report
        :rtype: str
        """
        # NEW FOR PROJECT 4a
        text = f'Documents: %i\n' % len(self)
        text += f'Sentences: %i\n' % sum([len(list(doc.sents)) for doc in self.get_documents()])
        entity_counts = self.get_entity_counts()
        text += f'Entities: %i\n' % sum([x[1] for x in entity_counts])
        text += f"Unique Entities: %i\n" % len(entity_counts)
        return text
        
    def get_noun_chunk_statistics(self):
        """Prints summary statistics for noun chunks in the corpus. Model on get_token_statistics.
        
        :returns: the statistics report
        :rtype: str
        """
        # NEW FOR PROJECT 4a
        text = f'Documents: %i\n' % len(self)
        text += f'Sentences: %i\n' % sum([len(list(doc.sents)) for doc in self.get_documents()])
        noun_counts = self.get_noun_chunk_counts()
        text += f'Noun Chunks:: %i\n' % sum([x[1] for x in noun_counts])
        text += f"Unique Noun Chunks: %i\n" % len(noun_counts)
        return text

    def get_basic_statistics(self):
        """Prints summary statistics for the corpus.
        
        :returns: the statistics report
        :rtype: str
        """
        # FOR PROJECT 4a: make this use get_token_statistics, get_entity_statistics and get_noun_chunk_statistics; also, instead of printing, return as a string.
        sentiment_counts={'positive':0, 'neutral': 0, 'negative':0}
        #count sentiments
        for doc in self.values():
            sentiment_label=doc['sentiment'][0]['label']
            if sentiment_label =='POSITIVE':
                sentiment_counts['positive']+=1
            elif sentiment_label=='NEGATIVE':
                sentiment_counts['negative']+=1
            else:
                sentiment_counts['neutral']+=1
        #add other statistics
        text=f'Documents: %i\n' %len(self)
        text+=f'Sentences: %i\n' %sum(len(list(doc.sents)) for doc in self.get_documents())
        text+=f'Positive Documents: {sentiment_counts["positive"]}\n'
        text+=f'Negative Document: {sentiment_counts["negative"]}\n'
        text+=f'Neutral Documents: {sentiment_counts["neutral"]}\n'
        text += f'{self.get_token_statistics()}\n'
        text+=f'{self.get_entity_statistics()}\n'
        text+=f'{self.get_noun_chunk_statistics()}'
        return text
    def plot_counts(self, counts, file_name):
        """Makes a bar chart for counts.

        :param counts: a list of item, count tuples
        :type counts: list
        :param file_name: where to save the plot
        :type file_name: string
        """
        fig = px.bar(x=[x[0] for x in counts], y=[x[1] for x in counts])
        fig.write_image(file_name)
    def get_sentiment_statistics(self):
        sentiment_counts={'positive': 0, 'neutral': 0, 'negative':0}
        #count sentiments
        for doc in self.values():
            sentiment_label=doc['sentiment'][0]['label']
            if sentiment_label =='POSITIVE':
                sentiment_counts['positive']+=1
            elif sentiment_label=='NEGATIVE':
                sentiment_counts['negative']+=1
            else:
                sentiment_counts['neutral']+=1
        return sentiment_counts
    def plot_token_frequencies(self, tags_to_exclude=['PUNCT', 'SPACE'], top_k=25):
        """Makes a bar chart for the top k most frequent tokens in the corpus.
        
        :param top_k: the number to keep
        :type top_k: int
        :param tags_to_exclude: tags to exclude
        :type tags_to_exclude: list[str]
        """
        #  FROM PROJECT 3c
         # Make a bar chart of the top most frequent tokens in the corpus (2 lines)
        # HINT: use the get_token_counts and plot_counts methods in corpus
        token_counts=self.get_token_counts(tags_to_exclude=tags_to_exclude, top_k=top_k)
        self.plot_counts(token_counts, 'token_counts.png')

    def plot_entity_frequencies(self, tags_to_exclude=['QUANTITY'], top_k=25):
        """Makes a bar chart for the top k most frequent entities in the corpus.
        
        :param top_k: the number to keep
        :type top_k: int
        :param tags_to_exclude: tags to exclude
        :type tags_to_exclude: list[str]
       """
        #  FROM PROJECT 3c
        # Make a bar chart of the top most frequent entities in the corpus (2 lines)
        reduced_entity_counts=counter(self.get_entity_counts(tags_to_exclude, top_k))
        self.plot_counts(reduced_entity_counts, 'entity_counts.png')
    def plot_keyphrase_frequencies(self):
        keyphrases=self.get_keyphrase_counts()
        self.plot_counts(keyphrases, top_k=10)
    def plot_noun_chunk_frequencies(self, top_k=25):
        """Makes a bar chart for the top k most frequent noun chunks in the corpus.
        
        :param top_k: the number to keep
        :type top_k: int
        """
        #  FROM PROJECT 3c
        # Make a bar chart of the top most frequent noun chunks in the corpus (2 lines)
        noun_chunks=self.get_noun_chunk_counts(top_k=top_k)
        self.plot_counts(noun_chunks, 'noun_chunk_frequencies.png')
     
    def plot_metadata_frequencies(self, key, top_k=25):
        """Makes a bar chart for the frequencies of values of a metadata key in a corpus.

        :param key: a metadata key
        :type key: str        
        :param top_k: the number to keep
        :type top_k: int
        """
        #  FROM PROJECT 3c
          # Make a bar chart of the top most frequent values for metadata key key (2 lines)
        metadata_counts=self.get_metadata_counts(key)
        self.plot_counts(metadata_counts, 'medata_counts.png')
 
    def plot_word_cloud(self, counts, name):
        """Plots a word cloud.

        :param counts: a list of item, count tuples
        :type counts: list
        :param file_name: where to save the plot
        :type file_name: string
        :returns: the word cloud
        :rtype: wordcloud
        """
        wc = wordcloud.WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(dict(counts))
        cloud = px.imshow(wc)
        cloud.update_xaxes(showticklabels=False)
        cloud.update_yaxes(showticklabels=False)
        cloud.write_html(f"{name}") 
        return cloud

    def plot_token_cloud(self, tags_to_exclude=['PUNCT', 'SPACE']):
        """Makes a word cloud for the frequencies of tokens in a corpus.

        :param tags_to_exclude: tags to exclude
        :type tags_to_exclude: list[str]
        :returns: the word cloud
        :rtype: wordcloud
        """
        #  FROM PROJECT 3c, then add return value
        token_counts=self.get_token_counts(tags_to_exclude=['PUNCT', 'SPACE'])
        token_cloud=self.plot_word_cloud(token_counts, 'token_cloud.png',)
        return token_cloud
 
    def plot_entity_cloud(self, tags_to_exclude=['QUANTITY']):
        """Makes a word cloud for the frequencies of entities in a corpus.
 
        :param tags_to_exclude: tags to exclude
        :type tags_to_exclude: list[str]
        :returns: the word cloud
        :rtype: wordcloud
        """
        #  FROM PROJECT 3c, then add return value
        entity_plot=self.get_entity_counts(tags_to_exclude=['QUANTITY'])
        entity_word_cloud=self.plot_word_cloud(entity_plot, 'entity_cloud.png')
        return entity_word_cloud
    def plot_noun_chunk_cloud(self):
        """Makes a word cloud for the frequencies of noun chunks in a corpus.

        :returns: the word cloud
        :rtype: wordcloudx
        """
        #  FROM PROJECT 3c, then add return value
        noun_counts=self.get_noun_chunk_counts()
        noun_cloud=self.plot_word_cloud(noun_counts, 'noun_chunk_cloud.png')
        return noun_cloud
        
    def render_doc_markdown(self, doc_id):
        """Render a document as markdown. From project 2a. 

        :param doc_id: the id of a spaCy doc made from the text in the document
        :type doc: str
        :returns: the markdown
        :rtype: str

        """
        # MODIFIED FROM PROJECT 3c: instead of printing or saving the markdown, return it as a string
        doc = self.get_document(doc_id)
        # Same definition as in project 3b, but prefix the output file name with self.name to make it unique to this corpus
        # define 'text' and set the title to be the document key (file name)
        file_name=self.name
        text = '# ' + doc_id + '\n\n'
        # walk over the tokens in the document
        for token in doc:
        # if the token is a noun, add it to 'text' and make it boldface (HTML: <b> at the start, </b> at the end)
            if token.pos_ == 'NOUN':
                text = '**' + text + '**' + token.text + '**'
        # otherwise, if it's a verb, add it to 'text' and make it italicized (HTML: <i> at the start, </i> at the end)
            elif token.pos_ == 'VERB':
                text = '*' + text + '*' + token.text + '*'
        # otherwise, just add it to 'text'!
            else:
                text = text + token.text
        # add any whitespace following the token using attribute whitespace_
            text = text + token.whitespace_
        # open an output file, named after the document with _text.md appended
        with open(self.name + '_' + doc_id.replace('/', '') + '_text.md', 'w') as outf:
            # write 'text'
            outf.write(text)
        return text

    def render_doc_table(self, doc_id):
        """Render a document's token and entity annotations as a table. From project 2a. 

        :param doc_id: the id of a spaCy doc made from the text in the document
        :type doc: str
        :returns: the markdown
        :rtype: str
        """
        # MODIFIED FROM PROJECT 3c: instead of printing or saving the markdown, return it as a string
        doc = self.get_document(doc_id)
        # Same definition as in project 3b, but prefix the output file name with self.name to make it unique to this corpus
        # make the tokens table
        tokens_table = "| Tokens | Lemmas | Coarse | Fine | Shapes | Morphology |\n| ------ | ------ | ------ | ---- | ------ | ---------- | \n"
        # walk over the tokens in the document
        # walk over the tokens in the document
        for token in doc:
        # if the token's part of speech is not 'SPACE'
            if token.pos_ != 'SPACE':
            # add the the text, lemma, coarse- and fine-grained parts of speech, word shape and morphology for this token to `tokens_table`
                tokens_table  = tokens_table + "| " + token.text + " | " + token.lemma_ + " | " + token.pos_ + " | " + token.tag_ + " | " + token.shape_ + " | " + re.sub(r'\|', '#', str(token.morph)) + " |\n"
    # Make the entities table
        entities_table = "| Text | Type |\n| ---- | ---- |\n"
    # walk over the entities in the document
        for entity in doc.ents:
        # add the text and label for this entity to 'entities_table'
            entities_table = entities_table + "| " + entity.text + " | " + entity.label_ + " |\n"
        return '## Tokens\n' + tokens_table + '\n## Entities\n' + entities_table

    def render_doc_statistics(self, doc_id):
        """Render a document's token and entity counts as a table. From project 2a. 

        :param doc_id: the id of a spaCy doc made from the text in the document
        :type doc: str
        :returns: the markdown
        :rtype: str
        """
        # MODIFIED FROM PROJECT 3c: instead of printing or saving the markdown, return it as a string
        doc = self.get_document(doc_id)
        # Same definition as in project 3b, but prefix the output file name with self.name to make it unique to this corpus
        # make a dictionary for the statistics
        stats={}
        text = '| Token/Entity | Count |\n | ------------ | ----- |\n'
        # print the key and count for each entry in 'stats'
        for token in doc:
        # if the token's part of speech is not 'SPACE'
            if token.pos_ != 'SPACE':
            # add the token and its part of speech tag ('pos_') to 'stats' (check if it is in 'stats' first!)
                if token.text + token.pos_ not in stats:
                    stats[token.text + token.pos_] = 0
            # increment its count
                stats[token.text + token.pos_] = stats[token.text + token.pos_] + 1
    # walk over the entities in the document
        for entity in doc.ents:
        # add the entity and its label ('label_') to 'stat's (check if it is in 'stat's first!)
            if entity.text + entity.label_ not in stats:
                stats[entity.text + entity.label_] = 0
        # increment its count
            stats[entity.text + entity.label_] = stats[entity.text + entity.label_] + 1
    # open an output file, named after the document with _stats.md appended
        # open an output file, named after the document with _stats.md appended
       # with open(self.name + '_' + doc_id.replace('/', '') + '_stats.md', 'w') as outf:
            # write the header for a table of tokens/entities and counts
            #outf.write('| Token/Entity | Count |\n | ------------ | ----- |\n')
            # print the key and count for each entry in 'stats'
        for key in sorted(stats.keys()):
            print('| ' + key + ' | ' + str(stats[key]) + ' |\n')
        return text
    def get_document_texts(self):
        """Takes an argument self and returns a list of pairs (id, text) corresponding to that argument
        :returns: a list
        :rtype:list"""
        return [(doc_id, doc['doc'].text) for doc_id, doc in self.items() if 'doc' in doc]
        # keys=self.keys()
        # results=[]
        # for key in keys:
        #     doctext=self[key]['doc'].text
        #     results.append((key,doctext))
        # return results

    def update_document_metadata(self, id, metadata):
        """Takes a document id and a dictionary of metadata key:value pairs, adds each key:value pair to the document in the corpus at the id, 
        and if there is no such document id in the corpus it should print an error message
        id (any): The unique identifier of the document.
        metadata (dict): Dictionary of metadata key:value pairs.
    Returns: dict: Updated corpus with modified document metadata."""
        if id in self:
            self[id]['metadata'] = metadata
        else:
            print('error')
        return self
    def get_topic_model(self):
        """Generates and visualizes a topic model for corpus documents using embeddings.

    Returns: str: Visualization of the generated topic model."""
        full_texts=[str(self[x]['doc'].text) for x in self]*50
        topic_model=self.build_topic_model()
        embeddings=self.embedding_model.encode(full_texts,show_progress_bar=True)
        topic_model.fit_transform(full_texts, embeddings)
        return topic_model.visualize_topics()
    def get_topic_model_document(self):
        """
    Generates a topic model and visualizes document embeddings using UMAP.

    Returns:
        str: Visualization of document embeddings in the generated topic model."""
        full_texts=[str(self[x]['doc'].text) for x in self]*50
        topic_model=self.build_topic_model()
        embeddings=self.embedding_model.encode(full_texts,show_progress_bar=True)
        topic_model.fit_transform(full_texts, embeddings)
        reduced_embeddings=UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
        return topic_model.visualize_documents(full_texts, reduced_embeddings=reduced_embeddings)
    @classmethod
    def build_topic_model(cls):
        """
    Builds a topic model using specified models for embedding, UMAP, HDBSCAN, vectorization, and representation.

    Returns:
        BERTopic: Initialized BERTopic model."""
        """Builds topic model"""
        representation_model={
            "KeyBERT": cls.keybert_model,
            "MMR": cls.mmr_model,
            "POS":cls.pos_model
        }
        topic_model=BERTopic(

            #Pipeline models
            embedding_model=cls.embedding_model,
            umap_model=cls.umap_model,
            hdbscan_model=cls.hdbscan_model,
            vectorizer_model=cls.vectorizer_model,
            representation_model=representation_model,
            #Hyperparameters
            top_n_words=10,
            verbose=True,
            nr_topics='auto'
            )
        
        return topic_model

    @classmethod
    def load_textfile(cls, file_name, my_corpus=None):
        """Loads a textfile into a corpus.

        :param file_name: the path to a text file
        :type file_name: string
        :param my_corpus: a corpus
        :type my_corpus: corpus
        :returns: a corpus
        :rtype: corpus
         """
        #  FROM PROJECT 3c
        if my_corpus == None:
            my_corpus = corpus()

    @classmethod  
    def load_jsonl(cls, file_name, my_corpus=None):
        """Loads a jsonl file into a corpus.

        :param file_name: the path to a jsonl file
        :type file_name: string
        :param my_corpus: a my_corpus
        :type my_corpus: my_corpus
        :returns: a my_corpus
        :rtype: my_corpus
         """
        #  FROM PROJECT #c
        # make sure we have a my_corpus
        if my_corpus == None:
            my_corpus = corpus()
        # Most the same as in project 3b, but use the corpus add method; don't forget to return my_corpus (6 lines of code)
        # open file_name as f
        with open(file_name, encoding='utf-8') as f:
        # walk over all the lines in the file
            for line in f.readlines():
                # load the python dictionary from the line using the json package; assign the result to the variabe 'js'
                    js=json.loads(line)
                 #take this line out when you have filled in the code below
                # if there are keys 'id' and 'fullText' in 'js'
                    if 'id' in js and 'fullText' in js: 
                        #my_corpus[js["id"]] = {'metadata': js, 'doc': cls.nlp(''.join(js["fullText"]))}
                        my_corpus.add_document(js["id"],''.join(js["fullText"]), js)
        return my_corpus

    @classmethod   
    def load_compressed(cls, file_name, my_corpus=None):
        """Loads a zipfile into a corpus.

        :param file_name: the path to a zipfile
        :type file_name: string
        :param my_corpus: a corpus
        :type my_corpus: corpus
        :returns: a corpus
        :rtype: corpus
       """
        #  FROM PROJECT #c
        # make sure we have a corpus
        if my_corpus == None:
            my_corpus = corpus()
        # Mostly the same as in project 3b; don't forget to return my_corpus (5 lines of code)
        shutil.unpack_archive(file_name, 'temp')
    # for each file_name in the compressed file
        for file_name2 in glob.glob('temp/*'):
         # take this line out when you have filled in the code below
        # build the corpus using the contents of file_name2
            cls.build_corpus(file_name2,my_corpus) 
    # clean up by removing the extracted files
        shutil.rmtree("temp")
        return my_corpus

    @classmethod
    def build_corpus(cls, pattern, my_corpus=None):
        """Builds a corpus from a pattern that matches one or more compressed or text files.

        :param pattern: the pattern to match to find files to add to the corpus
        :type file_name: string
        :param my_corpus: a corpus
        :type my_corpus: corpus
        :returns: a corpus
        :rtype: corpus
         """
        #  FROM PROJECT 3c
             # make sure we have a corpus
        if my_corpus == None:
            my_corpus = corpus(pattern)
       # Mostly the same as in project 3b; don't forget to return my_corpus (11 lines of code)
        #try:
         # take this line out when you have filled in the code below
        # for each file_name matching pattern
        for file_name in glob.glob(pattern):
        #if file_name ends with '.zip', '.tar' or '.tgz'
            if file_name.endswith('.zip') or file_name.endswith('.tar') or file_name.endswith('.tgz'):
                # then call load_compressed
                cls.load_compressed(file_name,my_corpus)
            # otherwise (we assume the files are just text)
            elif file_name.endswith('.jsonl'):
                cls.load_jsonl(file_name,my_corpus)
            else:
                # then call load_textfile
                cls.load_textfile(file_name,my_corpus)
        #except Exception as e: # if it doesn't work, say why
            #print(f"Couldn't load % s due to error %s" % (pattern, str(e)))
    # return the corpus
        return my_corpus