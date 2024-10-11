import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import hnswlib

import streamlit as st

path = 'All Appliances.csv'
model = SentenceTransformer('all-MiniLM-L6-v2')


def read_data(path):
    data = pd.read_csv(path)
    data['combined_features'] = data['name'] + ' ' + \
        data['main_category'] + ' '+data['sub_category']
    return data


def encoding(data, model=model, filter=''):
    sentences = data['combined_features'].tolist()
    embeddings = model.encode(sentences)
    print("encoding data done")
    return embeddings


def recommend_new_product_similarities(new_product_description, hnsw_index, k=10):
    # Create an embedding for the new product description
    new_product_embedding = model.encode([new_product_description])[0]

    # Find top-k similar items
    labels, distances = hnsw_index.knn_query(new_product_embedding, k=k)
    similar_indices = labels.flatten()

    return similar_indices


def main():
    # Function to display recommendations based on category
    def display_recommendations(category, search_query, output):
        # Display only the top 10 results
        st.write(
            f"Recommendations for {category} with search: '{search_query}'")
        st.write(output)

    # App Title
    st.title("Recommendation System")

    # Sidebar
    st.sidebar.header("Category")

    # Categories for filtering
    categories = ['appliances', 'car & motorbike', 'tv, audio & cameras',
                  'sports & fitness', 'grocery & gourmet foods', 'home & kitchen',
                  'pet supplies', 'stores', 'toys & baby products', "kids' fashion",
                  'bags & luggage', 'accessories', "women's shoes",
                  'beauty & health', "men's shoes", "women's clothing",
                  'industrial supplies', "men's clothing", 'music',
                  'home, kitchen, pets']

    selected_category = st.sidebar.radio("Select Category", categories)

    # Main page
    st.subheader("Search Recommendations")

    # Search bar
    search_query = st.text_input("Enter your search here")

    # Home button
    if st.button("Home"):
        st.write("Welcome to the Home Page!")

    # Read data
    data = read_data(path)
    embeddings = np.load('data.npy')  # load
    filtered_data = data[data['main_category'] == selected_category]
    sentences = filtered_data['combined_features'].to_list()

    dimension = embeddings.shape[1]
    p = hnswlib.Index(space='cosine', dim=dimension)
    p.init_index(max_elements=10000, ef_construction=200, M=16)
    p.add_items(embeddings)
    p.set_ef(50)  # ef should always be > k
    new_sentence = search_query
    new_embedding = model.encode([new_sentence])
    # Fetch k neighbors
    labels, distances = p.knn_query(new_embedding, k=1)

    output = sentences[labels[0][0]]
    # Display recommendations when search query or category is selected
    if search_query or selected_category:
        display_recommendations(selected_category, search_query, output)


if __name__ == "__main__":
    st.set_page_config(page_title="Recommended System",
                       page_icon="🎈", layout="wide")
    main()
