import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import pandas as pd
import json  # <--- IMPORT JSON

# --- Functions (unchanged) ---

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


def fetch_page_text(url: str) -> str | None:
    # ... (code unchanged) ...
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
            print(f"[{url}] Error: Status {response.status_code}")
            return f"Error: Status {response.status_code}"
    except requests.exceptions.RequestException as e:
        print(f"[{url}] Request error: {e}")
        return f"Request error: {e}"


@st.cache_resource
def load_hf_pipeline(model_path: str):
    # ... (code unchanged, this function is correct) ...
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Loading model onto device: {device}")
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
    # ... (code unchanged) ...
    try:
        results = ner_pipeline(text)
        products = []
        for entity in results:
            if entity['entity_group'] == 'PRODUCT':
                products.append(entity['word'])
        unique_products = list(dict.fromkeys(products))
        return unique_products
    except Exception as e:
        print(f"Error analyzing text (possibly too long): {e}")
        return ["! ANALYSIS ERROR !"]


# --- CONSTANTS (unchanged) ---
STR_FAILED = "--- (Failed to load page) ---"
STR_NO_PRODUCTS = "--- (No products found) ---"
STR_ERROR = "! ANALYSIS ERROR !"


@st.cache_data
def process_all_urls(csv_path: str, model_path: str):
    # ... (code unchanged) ...
    try:
        ner_pipeline = load_hf_pipeline(model_path)
    except Exception as e:
        st.error(f"Failed to load model from {model_path}. Did you train it?")
        st.exception(e)
        return None
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"File {csv_path} not found! Please make sure it's in the same folder.")
        return None
    if 'max(page)' not in df.columns:
        st.error(f"Column 'max(page)' not found in {csv_path}")
        return None
    urls_to_process = df['max(page)'].dropna().unique()
    total_urls = len(urls_to_process)
    results_list = []
    st.info(f"Starting analysis of {total_urls} URLs... This may take a long time.")
    progress_bar = st.progress(0, text="Initializing...")
    for i, url in enumerate(urls_to_process):
        progress_text = f"[{i + 1}/{total_urls}] Analyzing: {url[:70]}..."
        progress_bar.progress((i + 1) / total_urls, text=progress_text)
        text_to_analyze = fetch_page_text(url)
        if not text_to_analyze or text_to_analyze.startswith("Error"):
            products_list_str = STR_FAILED
        else:
            products_list = extract_products(text_to_analyze, ner_pipeline)
            if not products_list:
                products_list_str = STR_NO_PRODUCTS
            elif products_list == [STR_ERROR]:
                products_list_str = STR_ERROR
            else:
                products_list_str = " • ".join(products_list)
        results_list.append({
            "URL": url,
            "Found Products": products_list_str
        })
    progress_bar.empty()
    return pd.DataFrame(results_list)


# --- APPLICATION INTERFACE ---

st.set_page_config(layout="wide")
st.title("🛍️ Batch Product Analyzer")
st.markdown("This tool will analyze **all** URLs from the `URL_list.csv` file and show the results.")

MODEL_PATH = "../ner_model_transformers_EPOCHS10_BATCHES16"
CSV_PATH = "../URL_list.csv"
METRICS_FILE_PATH = "product_extraction_results_5_16.json"  # <-- DEFINE METRICS FILENAME

if st.button(f"🚀 Start Full Analysis of {CSV_PATH}"):

    results_df = process_all_urls(CSV_PATH, MODEL_PATH)

    if results_df is not None:
        st.success("✅ Analysis complete!")

        # --- Metrics Section (unchanged) ---
        is_failed = results_df["Found Products"].isin([STR_FAILED, STR_ERROR])
        is_no_products = results_df["Found Products"] == STR_NO_PRODUCTS
        is_found = ~is_failed & ~is_no_products
        found_products_df = results_df[is_found]
        num_total_urls = len(results_df)
        num_failed = len(results_df[is_failed])
        num_working_pages = num_total_urls - num_failed
        num_products_found = len(found_products_df)
        success_rate = 0.0
        if num_working_pages > 0:
            success_rate = (num_products_found / num_working_pages) * 100
        st.markdown("---")
        st.subheader("Analysis Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total URLs Analyzed", num_total_urls)
        col2.metric("Working Pages (Loaded OK)", num_working_pages)
        col3.metric("Pages with Products Found", num_products_found)
        st.metric(
            label="Product Finding Success Rate",
            value=f"{success_rate:.2f}%",
            help="The percentage of *working* pages (those that loaded successfully) where at least one product was found."
        )
        st.markdown("---")

        # --- START: NEW CODE TO SAVE METRICS ---
        metrics_data = {
            "total_urls_analyzed": num_total_urls,
            "working_pages": num_working_pages,
            "pages_with_products_found": num_products_found,
            "success_rate_percent": success_rate
        }
        try:
            with open(METRICS_FILE_PATH, 'w') as f:
                json.dump(metrics_data, f, indent=4)
            st.success(f"Metrics saved to {METRICS_FILE_PATH}")
        except Exception as e:
            st.error(f"Failed to save metrics file: {e}")
        # --- END: NEW CODE TO SAVE METRICS ---

        # --- Display tables (unchanged) ---
        st.markdown(f"### Pages Where Products Were Found ({num_products_found} URLs):")
        st.dataframe(found_products_df, width='stretch', height=300)
        st.markdown(f"### Full Analysis Results (All {num_total_urls} URLs):")
        st.dataframe(results_df, width='stretch', height=500)


        # --- Download button (unchanged) ---
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')


        csv_data = convert_df_to_csv(results_df)
        st.download_button(
            label="📥 Download Full Results as CSV",
            data=csv_data,
            file_name="product_extraction_results_5_16.csv",
            mime="text/csv",
        )
else:
    st.info("Click the button to start the analysis.")