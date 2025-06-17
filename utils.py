import pandas as pd
from Bio import Entrez
#from google.colab import drive
import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import time
from datetime import datetime
from itertools import combinations

domain_keywords = {
    'Retrieval-Augmented Generation': 'rag',
    'Retrieval Augmented Generation': 'rag', 
    'Knowledge Graph': 'KG',
    'Knowledge-Graph': 'KG',
    'semantic network': 'KG',
    'linked network': 'KG',
    'gnn': 'KG',
    'graph': 'graph',
    'Graph Database': 'KG',
    'neo4j': 'KG',
    'Cancer': 'cancer',
    'onco': 'cancer',
    'Agentic': 'agentic',
    'Multiagent Systems': 'MAS',
    'Multi-agent Systems': 'MAS',
    'Chatbot': 'chatbot',
    'Conversational AI': 'conversational AI',
    'qa':'qa',
    'multimodal':'mm',
    'multi-modal':'mm',
    'vlms':'vlms',
    'name entity recognition':'ner',
    'name entity':'ner',
    'name recognition':'ner',
    'entity recognition':'ner',
    'entity identification':'ner',
    'information extraction':'ner',
    'sequence labeling':'ner',
    'uml':'ner',
    'ulm':'ner',
    'speech':'speech',
    'speech recognition':'speech',
    'nlp':'nlp',
    'natural language processing':'nlp',
    'diagno':'diag',
    'review':'rev',
    'survei':'sur',
    'survey':'sur',
    'chatgpt':'cllm',
    'openai':'cllm',
    'gemini':'cllm',
    'claude':'cllm',
    'mistral':'cllm',
    'llama':'osllm',
    'deepseek':'osllm',
    'deepseek':'osllm',
    'finetune':'finetune',
    'finetun':'finetune',
    'fine-tune':'finetune',
    'rag':'rag',
    'agents':'agents',
    'patient care':'patient care',
    'patient monitoring':'patient monitoring',
    'imaging':'image',
    'image':'image',
    'img':'image',
    'decision support':'decision support',
    'decision':'decision support',
    'evaluation':'eval',
    'treatment':'treatment',
    'question answering':'qa',
    'question-answering':'qa',
    'question and answering':'qa',
    'hallucination':'hallucination',
    'large language model':'llm',
    'llm':'llm',
    'foundation model':'llm',
    'LLM':'llm',
    'LLMs':'llm',
    'LLMs':'llm',
    'LLM':'llm',
    'LLM':'llm',
    'heatlthcare':'domain',
    'medicine':'domain',
    'medical':'domain',
    'finance':'neg',
    'mental':'neg',
    'pshi':'neg',



}


def search_pubmed_and_save_csv(query, start_year, end_year, drive_folder_name="PubMed_Results"):
    """
   script para la busqueda masiva de papers en pubmed por criterios.

    Args:
        query (str): The search query for PubMed.
        start_year (int): The starting publication year (inclusive).
        end_year (int): The ending publication year (inclusive).
        drive_folder_name (str): The name of the folder in Google Drive to save the CSV.
                                 If it doesn't exist, it will be created.
    """
    Entrez.email = "YOUR_EMAIL@example.com"  #

 

    date_query = f'("{start_year}/01/01"[PDAT] : "{end_year}/12/31"[PDAT])'
    full_query = f'{query} AND {date_query}'


    handle = Entrez.esearch(db="pubmed", term=full_query, retmax="100000") # Set retmax to a sufficiently large number
    record = Entrez.read(handle)
    handle.close()


    id_list = record.get("IdList", [])
    print(f"Obtenidos {len(id_list)} resultados para consulta: {full_query}")

    if not id_list:
        print("Sin resultados")
        return

    # Fetch details for each paper ID
    fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="xml")
    papers = Entrez.read(fetch_handle)
    fetch_handle.close()

    # Extract desired fields into a list of dictionaries
    data = []
    # Check if 'PubmedArticle' exists and is a list
    if "PubmedArticle" in papers and isinstance(papers["PubmedArticle"], list):
        for paper in papers["PubmedArticle"]:
            # Initialize variables to default values
            pubmed_id = ""
            title = ""
            abstract = ""
            journal = ""
            publication_date = ""
            authors_str = ""
            mesh_terms_str = "" # Initialize mesh_terms_str
            keywords_str = "" # Initialize keywords_str
            article_type = "" # Initialize article_type
            volume = ""
            issue = ""
            pages = ""
            doi = ""


            if "MedlineCitation" in paper and isinstance(paper["MedlineCitation"], dict):
                medline_citation = paper["MedlineCitation"]
                if "Article" in medline_citation and isinstance(medline_citation["Article"], dict):
                    article = medline_citation["Article"]

                    pmid_element = medline_citation.get("PMID")
                    if isinstance(pmid_element, str):
                        pubmed_id = pmid_element
                    elif isinstance(pmid_element, dict):
                         pubmed_id = pmid_element.get("value", "")
                    else:
                        pubmed_id = "" # Default if PMID is neither string nor dict


                    title = article.get("ArticleTitle", "")

                    abstract = ""
                    if "Abstract" in article and isinstance(article["Abstract"], dict) and "AbstractText" in article["Abstract"]:
                        abstract_parts = article["Abstract"]["AbstractText"]
                        if isinstance(abstract_parts, list):
                            abstract = " ".join([part.get("value", "") for part in abstract_parts if isinstance(part, dict)])
                        elif isinstance(abstract_parts, dict):
                             abstract = abstract_parts.get("value", "")
                        elif isinstance(abstract_parts, str): # Handle cases where AbstractText is a string
                             abstract = abstract_parts


                    journal = ""
                    if "Journal" in article and isinstance(article["Journal"], dict):
                        journal = article["Journal"].get("Title", "")

                    publication_date = ""
                    if "Journal" in article and isinstance(article["Journal"], dict) and "JournalIssue" in article["Journal"]["JournalIssue"] and isinstance(article["Journal"]["JournalIssue"], dict) and "PubDate" in article["Journal"]["JournalIssue"] and isinstance(article["Journal"]["JournalIssue"]["PubDate"], dict):
                        pub_date = article["Journal"]["JournalIssue"]["PubDate"]
                        if "Year" in pub_date:
                            publication_date = pub_date["Year"]
                        elif "MedlineDate" in pub_date:
                             publication_date = pub_date["MedlineDate"]


                    authors = []
                    if "AuthorList" in article and isinstance(article["AuthorList"], list):
                        for author in article["AuthorList"]:
                            if isinstance(author, dict): # Ensure it's a dictionary
                                last_name = author.get("LastName", "")
                                fore_name = author.get("ForeName", "")
                                authors.append(f"{last_name}, {fore_name}".strip(", "))
                        authors_str = "; ".join(authors)

                    mesh_terms = []
                    if "MeshHeadingList" in medline_citation and isinstance(medline_citation["MeshHeadingList"], list):
                        for mesh_heading in medline_citation["MeshHeadingList"]:
                            if isinstance(mesh_heading, dict) and "DescriptorName" in mesh_heading and isinstance(mesh_heading["DescriptorName"], dict):
                                descriptor_name = mesh_heading["DescriptorName"].get("value", "")
                                mesh_terms.append(descriptor_name)
                        mesh_terms_str = "; ".join(mesh_terms)

                    keywords = []
                    if "KeywordList" in medline_citation and isinstance(medline_citation["KeywordList"], list):
                        for kw_list in medline_citation["KeywordList"]:
                            if isinstance(kw_list, dict) and "Keyword" in kw_list and isinstance(kw_list["Keyword"], list):
                                for keyword in kw_list["Keyword"]:
                                     if isinstance(keyword, dict):
                                        keywords.append(keyword.get("value", ""))
                                     elif isinstance(keyword, str): # Handle cases where keyword is a string
                                         keywords.append(keyword)
                        keywords_str = "; ".join(keywords)


                    article_type = "; ".join([t.get("value", "") for t in article.get("PublicationTypeList", []) if isinstance(t, dict)]) # Add check for dictionary type


                    # Add other fields you want to include with checks
                    volume = article.get("Journal", {}).get("JournalIssue", {}).get("Volume", "") if isinstance(article.get("Journal", {}).get("JournalIssue", {}), dict) else ""
                    issue = article.get("Journal", {}).get("JournalIssue", {}).get("Issue", "") if isinstance(article.get("Journal", {}).get("JournalIssue", {}), dict) else ""
                    pages = article.get("Pagination", {}).get("MedlinePgn", "") if isinstance(article.get("Pagination", {}), dict) else ""
                    doi = next((el.get("value", "") for el in article.get("ELocationID", []) if isinstance(el, dict) and el.get("ElocationIDType") == "doi"), "")


            data.append({
                "PubMed ID": pubmed_id,
                "title": title,
                "summary": abstract,
                "Journal": journal,
                "Publication Date": publication_date,
                "authors": authors_str,
                "MeSH Terms": mesh_terms_str,
                "Keywords": keywords_str,
                "Article Type": article_type,
                "Volume": volume,
                "Issue": issue,
                "Pages": pages,
                "DOI": doi,
                "query": query,
            })

    if data:
        df = pd.DataFrame(data)
        return df
    else:
        print("No data")
        return None



def search_arxiv(query, max_results=100, start=0, sort_by='relevance', date_range=None, retries=3, delay=5):
    """
    buscador papers en arxiv, devuelve los resultados en un panda.

    Args:
        query: consulta de búsqueda
        max_results: numero máximo de resultados
        start: Índice de inicio para paginación
        sort_by: criterio de ordenación , posibilidades('relevance', 'lastUpdatedDate', 'submittedDate')
        date_range:
        retries: Number of times to retry the request in case of connection errors
        delay: Initial delay in seconds between retries

    Returns:
        panda con los resultados
    """
    base_url = 'http://export.arxiv.org/api/query?'

    search_query = f'search_query=all:{query}&start={start}&max_results={max_results}&sortBy={sort_by}'
    full_url = base_url + search_query

    print(f"Consultando arXiv: {query}")

    for i in range(retries):
        try:
            response = requests.get(full_url)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            break # If the request is successful, break the loop
        except requests.exceptions.RequestException as e:
            print(f"Attempt {i+1} failed: {e}")
            if i < retries - 1:
                time.sleep(delay * (i + 1)) # Increase delay for subsequent retries
            else:
                print(f"Failed to retrieve data after {retries} attempts.")
                return pd.DataFrame()


    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return pd.DataFrame()

    root = ET.fromstring(response.content)

    ns = {'atom': 'http://www.w3.org/2005/Atom',
          'arxiv': 'http://arxiv.org/schemas/atom'}

    results = []
    for entry in root.findall('atom:entry', ns):

        paper = {
            'title': entry.find('atom:title', ns).text.strip().replace('\n', ' '),
            'summary': entry.find('atom:summary', ns).text.strip().replace('\n', ' '),
            'published': entry.find('atom:published', ns).text,
            'updated': entry.find('atom:updated', ns).text,
            'arxiv_url': entry.find('atom:id', ns).text,
            'pdf_url': None,
            'authors': [],
            'categories': [],
            'doi': None,
            'year': None
        }


        pub_date = paper['published']
        if pub_date:
            paper['year'] = int(pub_date[:4])  #  año en (YYYY-MM-DD)

        if date_range:
            year_start, year_end = date_range
            if paper['year'] < year_start or paper['year'] > year_end:
                continue

        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns).text
            authors.append(name)
        paper['authors'] = ', '.join(authors)


        categories = []
        for category in entry.findall('atom:category', ns):
            categories.append(category.get('term'))
        paper['categories'] = ', '.join(categories)

        for link in entry.findall('atom:link', ns):
            if link.get('title') == 'pdf':
                paper['pdf_url'] = link.get('href')

        arxiv_data = entry.find('arxiv:primary_category', ns)
        if arxiv_data is not None:
            paper['primary_category'] = arxiv_data.get('term')

        for link in entry.findall('atom:link', ns):
            if link.get('title') == 'doi':
                paper['doi'] = link.get('href')

        results.append(paper)
    df = pd.DataFrame(results)

    # Y-m-d
    for date_col in ['published', 'updated']:
        if date_col in df.columns and not df.empty:
            df[date_col] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
    df['query']=query
    return df

def search_and_export(query, max_results=100, year_start=None, year_end=None, filename=None):

    max_results = min(max_results, 2000)


    date_range = None
    if year_start or year_end:
        current_year = datetime.now().year
        year_start = year_start or 1991  # arXiv comenzó en 1991
        year_end = year_end or current_year
        date_range = (year_start, year_end)

    results = []
    batch_size = 500
    total_filtered = 0


    for start in range(0, max_results, batch_size):
        batch = min(batch_size, max_results - start)
        df_batch = search_arxiv(query, max_results=batch, start=start, date_range=date_range)

        if df_batch.empty:
            break

        results.append(df_batch)
        total_filtered += len(df_batch)
        print(f"Obtenidos {len(df_batch)} resultados en este lote (total: {total_filtered})")


        if start + batch_size < max_results:
            time.sleep(3)

    if not results:
        print("sin resultados")
        return None

    df_results = pd.concat(results, ignore_index=True)
    print(f"Totalresultados: {len(df_results)}")



    return df_results

def df_to_latex_with_integers(df, filename=None, caption=None, label=None):
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df_copy[col] = df_copy[col].fillna(0).astype(int)
    latex_table = df_copy.to_latex(index=False, escape=False)
    latex_table = latex_table.replace('\\begin{tabular}', '\\begin{tabular}{|' + '|'.join(['c'] * len(df_copy.columns)) + '|}')
    latex_table = latex_table.replace('\\end{tabular}', '\\hline\n\\end{tabular}')
    latex_table = latex_table.replace('\\toprule', '\\hline\\hline')
    latex_table = latex_table.replace('\\midrule', '\\hline\\hline')
    latex_table = latex_table.replace('\\bottomrule', '\\hline')
    header_line = latex_table.split('\n')[2]  # La línea con los encabezados
    bold_header = ' & '.join(['\\textbf{' + col + '}' for col in df_copy.columns])
    latex_table = latex_table.replace(header_line, bold_header + ' \\\\')
    
    if caption or label:
        latex_table = latex_table.replace('\\begin{tabular}', 
                                         '\\begin{table}[htbp]\n\\centering\n' + 
                                         (f'\\caption{{{caption}}}\n' if caption else '') + 
                                         (f'\\label{{{label}}}\n' if label else '') + 
                                         '\\begin{tabular}')
        latex_table = latex_table.replace('\\end{tabular}', '\\end{tabular}\n\\end{table}')
    
    if filename:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(latex_table)
    
    return latex_table



def detect_keywords(row, keyword_dict):
    """
    detecta keywords sobre columna "title" y "summary".

    Args:
        row: la fila del panda que tiene las keywords separadas por comas.
        keyword_dict: un diccionario con todas la relacion tags/keywords.

    Returns:
        .
    """
    keywords_found = set() # Use a set to avoid duplicate keywords

    title = str(row.get('title', '')).lower()
    summary = str(row.get('summary', '')).lower()

    text_to_search = title + " " + summary

    for key, keyword_value in keyword_dict.items():
        if text_to_search.find(key.lower()) != -1:
            keywords_found.add(keyword_value)

    return ', '.join(sorted(list(keywords_found)))



def concurrence_matriz_keywords(df, columna_keywords='keywords'):
    """
    matriz de co-ocurrencia para las palabras clave que aparecen
    en la misma fila de un DataFrame.


    Args:
        df (pd.DataFrame): el panmda.
        columna_keywords (str): columna de referencia.

    Returns:
        panda: Una matriz de co-ocurrencia.
.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El primer argumento 'df' debe ser un DataFrame de Pandas.")
    if columna_keywords not in df.columns:
        raise KeyError(f"La columna '{columna_keywords}' no se encuentra en el DataFrame.")
    if df.empty:
        return pd.DataFrame(dtype=int)


    listas_keywords_unicas_por_fila = []
    for keywords_str_fila in df[columna_keywords].fillna('').astype(str):
        palabras_clave_limpias = {kw.strip() for kw in keywords_str_fila.split(',') if kw.strip()}
        if palabras_clave_limpias: # Solo añadir si la fila tiene keywords válidas
            listas_keywords_unicas_por_fila.append(list(palabras_clave_limpias))

    if not listas_keywords_unicas_por_fila: 
        return pd.DataFrame(dtype=int)

    todas_las_keywords_flat = [kw for sublist in listas_keywords_unicas_por_fila for kw in sublist]
    
    if not todas_las_keywords_flat: 
        return pd.DataFrame(dtype=int)
        
    keywords_unicas_global = sorted(list(set(todas_las_keywords_flat)))

    matriz_coocurrencia = pd.DataFrame(0, index=keywords_unicas_global, columns=keywords_unicas_global, dtype=int)

    for kws_unicas_en_fila in listas_keywords_unicas_por_fila:
        for kw in kws_unicas_en_fila:
            matriz_coocurrencia.loc[kw, kw] += 1

        for kw1, kw2 in combinations(sorted(kws_unicas_en_fila), 2): 
            matriz_coocurrencia.loc[kw1, kw2] += 1
            matriz_coocurrencia.loc[kw2, kw1] += 1 

    return matriz_coocurrencia


def detect_keywords(row, keyword_dict):
    """
    Detects occurrences of dictionary keys in the 'title' and 'summary' columns
    of a DataFrame row and returns the corresponding values.

    Args:
        row: A pandas Series representing a row of the DataFrame.
        keyword_dict: A dictionary where keys are strings to search for and
                      values are the keywords to return if found.

    Returns:
        A comma-separated string of keywords found in the row's title or summary.
    """
    keywords_found = set() # Use a set to avoid duplicate keywords

    title = str(row.get('title', '')).lower()
    summary = str(row.get('summary', '')).lower()

    # Combine text from title and summary for searching
    text_to_search = title + " " + summary

    for key, keyword_value in keyword_dict.items():
        # Use find() for case-insensitive search (after converting to lower)
        if text_to_search.find(key.lower()) != -1:
            keywords_found.add(keyword_value)

    # Return keywords as a comma-separated string, sorted for consistency
    return ', '.join(sorted(list(keywords_found)))




def parse_wos_file(filepath):
    """
    parseo wos

    Args:
        filepath: .txt.

    Returns:
       panda parseadp.
    """

    # mapeo de etiquetas WOS a nombres de columnas
    wos_tag_mapping = {
        'PT': 'Publication Type',
        'AU': 'authors',
        'BA': 'Book Authors',
        'BF': 'Book First Name',
        'CA': 'Group Authors',
        'GP': 'Book Group Authors',
        'AF': 'Author Full Names',
        'SA': 'Suffixes',
        'EM': 'E-mail Addresses',
        'EP': 'Editors',
        'BP': 'Book Editors',
        'ED': 'Editors',
        'PU': 'Publisher',
        'PI': 'Publisher City',
        'PA': 'Publisher Address',
        'SC': 'Subject Category',
        'DE': 'Author Keywords',
        'ID': 'Keywords Plus',
        'AB': 'summary',
        'C1': 'Author Address',
        'RP': 'Reprint Address',
        'FX': 'Funding Details',
        'CR': 'Cited References',
        'NR': 'Cited Reference Count',
        'TC': 'Times Cited',
        'Z9': 'Total Times Cited', 
        'U1': 'Usage Count (Last 5 Years)',
        'U2': 'Usage Count (Since 2013)',
        'SO': 'Source Title',
        'SE': 'Series Title',
        'BS': 'Book Series Subtitle',
        'LA': 'Language',
        'DT': 'Document Type',
        'CT': 'Conference Title',
        'CY': 'Conference Date',
        'CL': 'Conference Location',
        'SP': 'Conference Sponsor',
        'J9': '29-Character Source Title',
        'JI': 'ISO Source Title',
        'PD': 'Publication Date',
        'PY': 'Publication Year',
        'VL': 'Volume',
        'IS': 'Issue',
        'PN': 'Part Number',
        'SU': 'Supplement',
        'SI': 'Special Issue',
        'BP': 'Beginning Page',
        'EP': 'Ending Page',
        'AR': 'Article Number',
        'PG': 'Page Count',
        'GA': 'Document Delivery Number',
        'UT': 'Unique Article Identifier', 
        'DA': 'Date Processed',
        'OA': 'Open Access Indicator',
        'TI': 'title',
        'SR': 'Source Research Area',
        'WC': 'Web of Science Categories',
        'UT': 'Accession Number', 
        'ER': 'End of Record', 
        'EF': 'End of File' 
    }

    records = []
    current_record = {}
    current_tag = None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue # 
                parts = line.split(' ', 1)
                if len(parts) > 1 and parts[0] in wos_tag_mapping:

                    tag = parts[0]
                    content = parts[1].strip()

                    if tag == 'PT' and current_record:

                        records.append(current_record)
                        current_record = {}

                    current_tag = wos_tag_mapping[tag]
                    current_record[current_tag] = content

                elif current_tag is not None and line:

                    current_record[current_tag] += ' ' + line


                if line == 'ER':
                    if current_record:
                        records.append(current_record)
                        current_record = {}
                        current_tag = None 


                if line == 'EF':
                    break


        if current_record:
             records.append(current_record)



        df = pd.DataFrame(records)

 
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()

        print(f"Successfully parsed {len(records)} records from {filepath}")
        return df

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return pd.DataFrame()

