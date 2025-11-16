import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import json
import os
import pandas as pd

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


def fetch_page_text(url: str) -> str | None:
    headers = {'User-Agent': USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            if soup.body:
                page_text = soup.body.get_text(separator=" ", strip=True)
            else:
                page_text = soup.get_text(separator=" ", strip=True)
            cleaned_text = " ".join(page_text.split())
            return cleaned_text
        else:
            print(f"Error (Status {response.status_code}) accessing {url}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request error for {url}: {e}")
        return None


@st.cache_resource
def load_hf_pipeline(model_path: str):
    """
    Loads the trained model and tokenizer from Hugging Face.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Loading model onto device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        add_prefix_space=True
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device
    )
    return ner_pipeline


def extract_products(text: str, ner_pipeline) -> list[str]:
    try:
        results = ner_pipeline(text)
        products = []
        for entity in results:
            if entity['entity_group'] == 'PRODUCT':
                products.append(entity['word'])
        unique_products = list(dict.fromkeys(products))
        return unique_products
    except Exception as e:
        st.error(f"Error during text analysis: {e}")
        st.warning("The text might be too long for the model. We could try splitting it.")
        return []


st.set_page_config(layout="wide")
st.title("🛍️ Furniture Product Extractor (Hugging Face)")

MODEL_PATH = "./ner_model_transformers"
METRICS_FILE_PATH = "metrics/batch_analysis_metrics.json"
CSV_FILE_PATH = "metrics/information_got_from_pages.csv"

tab1, tab2 = st.tabs(["Single URL Analyzer", "Batch Analysis Metrics"])

# --- TAB 1: SINGLE URL ANALYZER ---
with tab1:
    st.markdown("Test the model with a single URL.")

    try:
        pipeline = load_hf_pipeline(MODEL_PATH)
        st.success("NER model (Hugging Face) loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model from {MODEL_PATH}. Did you train it?")
        st.exception(e)
        st.stop()

    url = st.text_input("Enter the product page URL:",
                        "https://www.danishinspirations.com/products/products/bedroom/page/3/")

    if st.button("Extract Products"):
        if not url:
            st.warning("Please enter a URL.")
        else:
            with st.spinner(f"Analyzing {url}..."):
                # 1. Fetch text
                text_to_analyze = fetch_page_text(url)

                if not text_to_analyze or text_to_analyze.startswith("Error"):
                    st.error(f"Failed to get text: {text_to_analyze}")
                else:
                    # 2. Analyze
                    products = extract_products(text_to_analyze, pipeline)

                    # 3. Display results
                    st.subheader("Products Found:")
                    if products:
                        for prod in products:
                            st.markdown(f"- **{prod}**")
                    else:
                        st.warning("No products found on this page.")

                    with st.expander("View Extracted Text"):
                        st.text(text_to_analyze)

# TAB 2: BATCH ANALYSIS METRICS
with tab2:
    st.subheader("Latest Batch Analysis Metrics")

    if os.path.exists(METRICS_FILE_PATH):
        try:
            with open(METRICS_FILE_PATH, 'r') as f:
                metrics_data = json.load(f)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total URLs Analyzed", metrics_data.get("total_urls_analyzed", "N/A"))
            col2.metric("Working Pages", metrics_data.get("working_pages", "N/A"))
            col3.metric("Pages w/ Products Found", metrics_data.get("pages_with_products_found", "N/A"))

            rate = metrics_data.get("success_rate_percent", 0)
            st.metric(
                label="Product Finding Success Rate",
                value=f"{rate:.2f}%",
                help="The percentage of *working* pages where at least one product was found."
            )

        except Exception as e:
            st.error(f"Error reading metrics file: {e}")
            st.exception(e)
    else:
        st.warning(f"No metrics file found at `{METRICS_FILE_PATH}`.")
        st.info("Please run the `batch_app.py` script first to generate metrics.")

    # --- Display CSV data ---
    st.divider()  # Adds a horizontal line
    st.subheader("Batch Analysis Details")

    if os.path.exists(CSV_FILE_PATH):
        try:
            df = pd.read_csv(CSV_FILE_PATH, index_col=0)

            df.index.name = "ID"

            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Error reading CSV file ({CSV_FILE_PATH}): {e}")
            st.exception(e)
    else:
        st.warning(f"No CSV data file found at `{CSV_FILE_PATH}`.")
        st.info("This file is also generated by the `batch_app.py` script.")
