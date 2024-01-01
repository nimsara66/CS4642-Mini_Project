
import streamlit as st
import torch
import pandas as pd
import numpy as np
import json
import os
import uuid
import re

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch import helpers

from tqdm.auto import tqdm
tqdm.pandas()

st.set_page_config(page_title="Metaphor Mystique", layout="wide", initial_sidebar_state="expanded")

# UI text strings
page_title = "Metaphor Mystique"
page_helper = "Discover metaphors and meanings in ancient Sri Lankan poems"
empty_search_helper = "Select search terms and enter the metaphor in your own way"
category_list_header = "Search terms"
borough_search_header = "Select a borough"
term_search_header = "Select search terms"
semantic_search_header = "What metaphor are you looking for?"
semantic_search_placeholder = "මහ ගුණ මුහුදාණන්"
search_label = "Search metaphor in poems"
venue_list_header = "Venue details"

# Handler functions
def handler_load_searchterms():
    """
    Load search terms for the selected borough and update session state.
    """
    searchterms = [
        {'TERM': 'Poem Name'},
        {'TERM': 'Poet'},
        {'TERM': 'Poem'},
        {'TERM': 'Metaphorical Term'}
    ]
    st.session_state.searchterms_list = [term['TERM'] for term in searchterms]


def handler_search_metaphor():
    """
    Search for metaphor based on user query and update session state with results.
    """
    try:
        model = SentenceTransformer('Ransaka/sinhala-roberta-sentence-transformer')
        user_metaphor_query = st.session_state.user_metaphor_query
        user_metaphor_embeddings = model.encode([user_metaphor_query])
        encod_np_array = np.array(user_metaphor_embeddings)
        encod_list = encod_np_array.tolist()
        query_embeddings = encod_list[0]

        # Connect to elastic search
        try:
        #   ENDPOINT = "https://55d6-212-104-225-107.ngrok.io:443"
          ENDPOINT = "http://localhost:9200"
          USERNAME = "elastic"
          PASSWORD = "WRpT827K5LnRWFBUZh6r"

          es = Elasticsearch(hosts=[ENDPOINT],  http_auth=(USERNAME, PASSWORD), timeout=300)
          es.ping()
        except:
          st.error(f"⚠️ Elasticsearch connection failed at ENDPOINT: {ENDPOINT}.  \n  \nPlease try a different query.")

        # Build query
        knn_search_params = {
            "knn": {
                "field": "embeddings",
                "query_vector": query_embeddings,
                # @TODO: get from input
                "k": 50,
                "num_candidates": 1000
            },
            "_source": [
                "poem_name",
                "poet",
                "poem_number",
                "poem",
                "metaphorical_terms_si",
                "metaphorical_terms_en",
                "metaphorical_meaning_si",
                "metaphorical_meaning_en"
                ]
        }

        # Perform the KNN search
        query_results = es.knn_search(index="poem", body=knn_search_params)

        # Create a pandas dataframe from query results
        query_result_data = []
        for hit in query_results['hits']['hits']:
            _score = hit['_score']
            poem_name = hit['_source']["poem_name"]
            poet = hit['_source']["poet"]
            poem_number = hit['_source']["poem_number"]
            poem = hit['_source']["poem"]
            metaphorical_terms_si = hit['_source']["metaphorical_terms_si"]
            metaphorical_terms_en = hit['_source']["metaphorical_terms_en"]
            metaphorical_meaning_si = hit['_source']["metaphorical_meaning_si"]
            metaphorical_meaning_en = hit['_source']["metaphorical_meaning_en"]

            entry = {
                "_score": _score,
                "poem name": poem_name,
                "poet": poet,
                "poem number": poem_number,
                "poem": poem,
                "metaphorical terms sinhala": metaphorical_terms_si,
                "metaphorical terms english": metaphorical_terms_en,
                "metaphorical meaning sinhala": metaphorical_meaning_si,
                "metaphorical meaning english": metaphorical_meaning_en
            }

            query_result_data.append(entry)

        # Create a pandas DataFrame
        query_result_dataframe = pd.DataFrame(query_result_data)
        # query_result_dataframe.sort_values(by='_score', ascending=False)

        similarity_scores_terms = util.pytorch_cos_sim(query_embeddings, model.encode(query_result_dataframe['metaphorical terms sinhala'].tolist()))
        similarity_scores_poet = util.pytorch_cos_sim(query_embeddings, model.encode(query_result_dataframe['poet'].tolist()))
        similarity_scores_poem = util.pytorch_cos_sim(query_embeddings, model.encode(query_result_dataframe['poem'].tolist()))
        similarity_scores_poem_name = util.pytorch_cos_sim(query_embeddings, model.encode(query_result_dataframe['poem name'].tolist()))

        terms_score_multiplier = 1
        poem_score_multiplier = 1
        poet_score_multiplier = 1
        poem_name_score_multiplier = 1
        if "terms_selection" in st.session_state and len(st.session_state.terms_selection) > 0:
          if 'Poem Name' in st.session_state.terms_selection:
            poem_name_score_multiplier=3
          if 'Poet' in st.session_state.terms_selection:
            poet_score_multiplier=3
          if 'Poem' in st.session_state.terms_selection:
            poem_score_multiplier=3
          if 'Metaphorical Term' in st.session_state.terms_selection:
            terms_score_multiplier=3

        combined_scores = np.mean([
            terms_score_multiplier * similarity_scores_terms.cpu().numpy(),
            poet_score_multiplier * similarity_scores_poet.cpu().numpy(),
            poem_score_multiplier * similarity_scores_poem.cpu().numpy(),
            poem_name_score_multiplier * similarity_scores_poem_name.cpu().numpy()
            ], axis=0)

        query_result_dataframe['Combined_Score'] = combined_scores[0]
        retrieved_data = query_result_dataframe.sort_values(by='Combined_Score', ascending=False)
        st.session_state.suggested_metaphors = retrieved_data

    except Exception as e:
        st.error(f"{str(e)}")

# UI elements
def render_cta_link(url, label, font_awesome_icon):
    st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">', unsafe_allow_html=True)
    button_code = f'''<a href="{url}" target=_blank><i class="fa {font_awesome_icon}"></i> {label}</a>'''
    return st.markdown(button_code, unsafe_allow_html=True)

def render_search():
    """
    Render the search form in the sidebar.
    """
    search_disabled = True
    with st.sidebar:
        if "searchterms_list" in st.session_state and len(st.session_state.searchterms_list) > 0:
            st.multiselect(label=term_search_header, options=(st.session_state.searchterms_list), key="terms_selection", max_selections=3)

        st.text_input(label=semantic_search_header, placeholder=semantic_search_placeholder, key="user_metaphor_query")

        if "user_metaphor_query" in st.session_state and st.session_state.user_metaphor_query != "":
            search_disabled = False

        st.button(label=search_label, key="metaphor_search", disabled=search_disabled, on_click=handler_search_metaphor)

        st.write("---")
        render_cta_link(url="https://twitter.com/dclin", label="Let's connect", font_awesome_icon="fa-twitter")
        render_cta_link(url="https://linkedin.com/in/d2clin", label="Let's connect", font_awesome_icon="fa-linkedin")

def render_styles():
    styles = '''
        <style>
            * {
              box-sizing: border-box;
            }

            /* Custom CSS for the card component */
            .card {
                display: flex;
                border: 1px solid #333;
                box-shadow: 0 0 10px rgba(51, 51, 51, 0.2);
                padding: 0.5rem;
                border-radius: .1rem;
                background: rgba(17, 17, 17, 0.7);
            }

            .card-content {
                padding: 0 1rem;
                flex: 1;
            }

            .card-title {
                font-size: 1.5rem;
            }

            .card-subtitle {
                color: #ddd;
            }

            .card-media {
                display: flex;
                min-width: 40%;
                background-image: url('https://previews.123rf.com/images/peekeedee1/peekeedee11906/peekeedee1190600406/125248671-old-paper-texture-vintage-paper-background-or-texture-brown-paper-texture.jpg');
                background-size: cover;
                background-repeat: no-repeat;
                opacity: 0.7;
                color: #000;
                font-weight: bold;
                padding: 1rem;
                border-radius: .2rem;
                align-items: center;
                justify-content: center;
                margin: 1rem;
                border: 1px solid #aaa;
                box-shadow: 0 0 10px rgba(170, 170, 170, 0.2);

            }

            .meaning-content {
              padding-top: 2rem;
              align-items: center;
              word-wrap: break-word;
            }

            .meaning-content * {
              margin: 0;
              align-items: center;
            }

            .highlighted {
              background-color: yellow;
            }


            hr {
              box-shadow: 0 0 5px rgba(51, 51, 51, 0.2);
              margin: 0.6rem 0 !important;
            }
        </style>
    '''
    st.markdown(styles, unsafe_allow_html=True)

def render_card(poem_name, poet, poem, meaning_si, meaning_en):
    body = f'''
        <div class="card">
            <div class="card-content">
                <div class="card-title">{poem_name}</div>
                <div class="card-subtitle">{poet}</div>
                <div class="meaning-content">
                    <hr>
                    <p>{meaning_si}</p>
                    <hr>
                    <p>{meaning_en}</p>
                    <hr>
                </div>
            </div>
            <div class="card-media">
              <div>
                {poem}
              </div>
            </div>
        </div>
    '''
    st.markdown(body, unsafe_allow_html=True)
    st.write('\n')

def render_search_result():
    """
    Render the search results on the main content area.
    """
    # Number of entries per page
    entries_per_page = 5

    # Calculate the total number of pages
    total_pages = (len(st.session_state.suggested_metaphors) - 1) // entries_per_page + 1

    # Get the current page from the URL query parameter
    current_page = st.session_state.get('current_page', 1)

    # Create a paginated DataFrame
    start_idx = (current_page - 1) * entries_per_page
    end_idx = start_idx + entries_per_page
    paginated_suggested_metaphors = st.session_state.suggested_metaphors.iloc[start_idx:end_idx]

    render_styles()
    for index, row in paginated_suggested_metaphors.iterrows():
        render_card(
            row['poem name'],
            row['poet'],
            find_term_in_sentence(row['poem'].replace('\n', '<br>'), row["metaphorical terms sinhala"]),
            row['metaphorical meaning sinhala'], row['metaphorical meaning english']
        )

    # Pagination controls
    col1, col2= st.columns(2)

    with col1:
        if st.button("Previous Page", key='prev_page'):
            current_page = max(current_page - 1, 1)
    with col2:
        if st.button("Next Page", key='next_page'):
            current_page = min(current_page + 1, total_pages)

    # Store the current page in session state
    st.session_state.current_page = current_page

def find_term_in_sentence(sentence, term):
    # Replace "<br>"" with " <br>"
    sentence = re.sub(r'(<br>)', r' <br>', sentence)
    # Create a regex pattern that allows "<br>" anywhere in the term
    pattern = re.compile(r'(<br>|\s*)?'.join(map(re.escape, term)))

    # Search for the pattern in the sentence
    match = pattern.search(sentence)

    if match:
        matched_term =  match.group(0)
        return sentence.replace(matched_term, f'<span class="highlighted">{matched_term}</span>')
    else:
        return sentence

if "searchterms_list" not in st.session_state:
    handler_load_searchterms()
render_search()

st.title(page_title)
st.write(page_helper)
st.write("---")

if "suggested_metaphors" not in st.session_state:
    st.write(empty_search_helper)
else:
    render_search_result()
