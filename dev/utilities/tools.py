from duckduckgo_search import DDGS
# pip install duckduckgo_search

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