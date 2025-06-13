import pandas as pd
from Bio import Entrez
#from google.colab import drive
import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import time
from datetime import datetime


def search_pubmed_and_save_csv(query, start_year, end_year, drive_folder_name="PubMed_Results"):
    """
    Searches PubMed for papers based on a query and year range, retrieves all fields,
    and saves the results as a CSV file in a specified Google Drive folder.

    Args:
        query (str): The search query for PubMed.
        start_year (int): The starting publication year (inclusive).
        end_year (int): The ending publication year (inclusive).
        drive_folder_name (str): The name of the folder in Google Drive to save the CSV.
                                 If it doesn't exist, it will be created.
    """
    Entrez.email = "YOUR_EMAIL@example.com"  # Replace with your email address

 
    # Construct the date range query part
    date_query = f'("{start_year}/01/01"[PDAT] : "{end_year}/12/31"[PDAT])'
    full_query = f'{query} AND {date_query}'

    # Search PubMed
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
    buscador papers en arXiv y devuelve los resultados en un panda.

    Args:
        query: consulta de búsqueda
        max_results: numero máximo de resultados
        start: Índice de inicio para paginación
        sort_by: Criterio de ordenación ('relevance', 'lastUpdatedDate', 'submittedDate')
        date_range:
        retries: Number of times to retry the request in case of connection errors
        delay: Initial delay in seconds between retries

    Returns:
        DataFrame con los resultados
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