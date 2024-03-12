# import corpus from spacy_on_corpus
from spacy_on_corpus import corpus

# import anvil server
import anvil.server

# make a corpus instance called my_corpus
my_corpus=corpus.build_corpus('creator.jsonl')
#my_corpus = corpus()
anvil.server.connect("server_WI4ONLYTO7I5PBL7MVPFEJCT-4B4HCRS2P6XSSIZA")
# YOUR ANVIL CALLABLES HERE
# import anvil server, and connect using your API key
def run():
    """Run the server!"""  
    # connect
    #ganvil.server.connect("server_WI4ONLYTO7I5PBL7MVPFEJCT-4B4HCRS2P6XSSIZA")
    # wait forever
    anvil.server.wait_forever()
@anvil.server.callable
def load_file(filename, file_contents):
    """Call build_corpus on file_contents, giving it name filename
    
    :param filename: the filename we want to store file_contents in
    :type filename: str
    :param file_contents: the contents we want to use to build / augment my_corpus
    :type file_contents: byte stream
    """
    # first we write file_"contents to a file which will have name inputs/filename
    with open('inputs/' + filename, 'wb') as f:
      f.write(file_contents.get_bytes())
    # You call build_corpus on inputs/filename, giving it my_corpus as a keyword argument
    corpus.build_corpus('inputs/' + filename, my_corpus=my_corpus)
@anvil.server.callable
def add_document(text):
    """Add a document to my_corpus using contents.
    
    :param text: the text we want to add to my_corpus
    :type text: str
    """
    # You add a document to my_corpus using text and give it a unique id
    # HINT: try giving it an id corresponding to the size of my_corpus
    my_corpus.add_document(str(len(my_corpus)), text)
    print(my_corpus.keys())

@anvil.server.callable
def clear():
    """Empty my_corpus."""
    # You implement this using an instance method of dict
    my_corpus.clear()
@anvil.server.callable
def get_corpus_tokens_counts(top_k=25):
    """Get the token counts from my_corpus.
    
    :param top_k: the top_k tokens to return
    :type top_k: int
    :returns: a list of pairs (item, frequency)
    :rtype: list
    """
    # You return the token counts
    return my_corpus.get_token_counts()

@anvil.server.callable
def get_corpus_entities_counts(top_k=25):
    """Get the entity counts from my_corpus.
    
    :param top_k: the top_k entities to return
    :type top_k: int
    :returns: a list of pairs (item, frequency)
    :rtype: list
    """
    # You return the entity counts
    return my_corpus.get_entity_counts()

@anvil.server.callable
def get_corpus_noun_chunks_counts(top_k=25):
    """Get the noun chunk counts from my_corpus.
    
    :param top_k: the top_k noun chunks to return
    :type top_k: int
    :returns: a list of pairs (item, frequency)
    :rtype: list
    """
    # You return the noun chunk counts
    return my_corpus.get_noun_chunk_counts()
@anvil.server.callable
def get_corpus_tokens_statistics():
    """Get the token statistics from my_corpus.
    
    :returns: basic statistics suitable for printing
    :rtype: str
    """
    # You return the token statistics
    return my_corpus.get_token_statistics()

@anvil.server.callable
def get_corpus_entities_statistics():
    """Get the entity statistics from my_corpus.
    
    :returns: basic statistics suitable for printing
    :rtype: str
    """
    # You return the entity statistics
    return my_corpus.get_entity_statistics()

@anvil.server.callable
def get_corpus_noun_chunks_statistics():
    """Get the noun chunk statistics from my_corpus.
    
    :returns: basic statistics suitable for printing
    :rtype: str
    """
    # You return the noun chunk statistics
    return my_corpus.get_noun_chunk_statistics()
@anvil.server.callable
def get_corpus_keyphrase_statistics():
    return my_corpus.get_keyphrase_statistics()
@anvil.server.callable
def get_token_cloud():
    """Get the token cloud for my_corpus.
    
    :returns: an image
    :rtype: plot
    """
    # You get the token counts
    token_counts=my_corpus.get_token_counts()
    # You make the word cloud if token_counts is not None
    if token_counts!= None:
        return my_corpus.plot_token_cloud()
    else:
        return None

@anvil.server.callable
def get_entity_cloud():
    """Get the entity cloud for my_corpus.
    
    :returns: an image
    :rtype: plot
    """
    # You get the entity counts
    entity_counts=my_corpus.get_entity_counts()
    # You make the entity cloud if entity_counts is not None
    if entity_counts != None:
        return my_corpus.plot_entity_cloud()
    else:
        return None
@anvil.server.callable
def get_noun_chunk_cloud():
    """Get the noun chunk cloud for my_corpus.
    
    :returns: an image
    :rtype: plot
    """
    # You get the noun chunk counts
    noun_counts=my_corpus.get_noun_chunk_counts
    # You make the noun chunk cloud if chunk_counts is not None
    if noun_counts!=None:
        return my_corpus.plot_noun_chunk_cloud()
    else:
        return None

@anvil.server.callable
def get_document_ids():
    """Get the ids of all document ids in the corpus.
    
    :returns: the document ids
    :rtype: list[str]
    """
    # You get the list of document ids in the corpus
    return list(my_corpus.keys())

@anvil.server.callable
def get_doc_markdown(doc_id):
    """Get the document markdown for a document in my_corpus.
    
    :param doc_id: a document id
    :type doc_id: str
    :returns: markdown
    :rtype: str
    """
    # You do it!
    return my_corpus.render_doc_markdown(doc_id)
@anvil.server.callable
def get_doc_table(doc_id):
    """Get the document table for a document in my_corpus.
    
    :param doc_id: a document id
    :type doc_id: str
    :returns: markdown
    :rtype: str
    """
    # You do it!
    return my_corpus.render_doc_table(doc_id)

@anvil.server.callable
def get_doc_statistics(doc_id):
    """Get the document statistics for a document in my_corpus.
    
    :param doc_id: a document id
    :type doc_id: str
    :returns: markdown
    :rtype: str
    """
    # You do it!
    return my_corpus.render_doc_statistics(doc_id)
@anvil.server.callable
def get_corpus_statistics():
    """Gets the basic statistics of a corpus
    returns: text"""
    return my_corpus.get_basic_statistics()
@anvil.server.callable
def get_corpus_sentiment():
    """returns a dictionary with positive, negative and neutral sentiments for this corpus
    rtype: dict"""
    return my_corpus.get_sentiment_statistics()
@anvil.server.callable
def get_document_summary(doc_id): 
    """Returns the summary of a document in the document
    rtype: text"""
    summary=my_corpus[doc_id]['summary'][0]
    return summary['summary_text']
@anvil.server.callable
def get_topic_models_plot():
    """returns a topic plot for the corpus
    rtype: image"""
    return my_corpus.get_topic_model()
@anvil.server.callable
def get_topic_documents_plot():
    """returns a document plot for the corpus
    rtype: image"""
    return my_corpus.get_topic_model_document()
#@anvil.callable
def get_doc_sentiment_markdown(doc_id):
    """returns a markdown table of the sentences in a document with their sentiments"""
    return my_corpus.render_document_sentiments(doc_id)
# this says, if executing this on the command line like python server.py, run run()    
if __name__ == "__main__":
    run()