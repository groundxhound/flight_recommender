score_model = True
algorithm = "LSI"
authorization_code = "7c1f78ff1472050b902566e60a5b3d16f35e94ae3b52c03b32d3e95f19e563ef"
neo4j_url = "http://neo4j:neo4j@localhost:7474/db/data/"
photoapi_disabled = True
config_def = {'algorithm': 'LSI',  # Algorithm that will be used LSI or LDA
              'min_word_count': 8,  # All words that appear less often will be deleted in preprocessing
              'no_above_fraction': 0.6,  # If words appear in more than no_above_fraction percent of all documents they
              # will be ignored. 0.6 = 60%
              'delete_numbers': True,  # Will regex delete numbers from the text. So F1 -> F
              'delete_words': ['also'],  # specify additional words that you want to be ignored
              "nr_topics": 20,  # parameter for the algorithm
              "tfidf": True}  # term frequency inverse document frequency
create_topics = True
