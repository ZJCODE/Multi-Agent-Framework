from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
# pip install duckduckgo_search
# pip install requests
# pip install beautifulsoup4

def web_search(query:str,timelimit:str,max_results:int):
    """
    Search the web for the given query with the specified time limit and maximum number of results.

    Args:
        query (str): The query to search.
        timelimit (Literal['d', 'w', 'm', 'y']): The time limit for the search.
        max_results (int): The maximum number of results to return usually set to 10.

    """
    results = DDGS().text(keywords=query, safesearch='off', timelimit=timelimit, max_results=max_results)
    return results

def web_content_process(link: str) -> str:
    """
    Process the content of the given link by using beautifulsoup.

    Args:
        link (str): The link to process.

    Returns:
        str: The main content of the link.
    """
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Attempt to find the main content using common tags
    main_content = ''
    if soup.main:
        main_content = soup.main.get_text(strip=True)
    elif soup.article:
        main_content = soup.article.get_text(strip=True)
    elif soup.section:
        main_content = soup.section.get_text(strip=True)
    elif soup.body:
        main_content = soup.body.get_text(strip=True)
    else:
        main_content = soup.get_text(strip=True)

    return main_content
