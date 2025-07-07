import httpx
from agents import function_tool
from typing_extensions import TypedDict


class CitationQuery(TypedDict):
    query: str


@function_tool
async def citation_finder(input: CitationQuery) -> str:
    """
    Searches Semantic Scholar for academic papers related to the query.
    """
    query = input["query"]
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": 1, "fields": "title,authors,url"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            papers = data.get("data", [])
            if not papers:
                return "No academic citations found."

            top_paper = papers[0]
            title = top_paper.get("title", "No title")
            authors = ", ".join(a["name"] for a in top_paper.get("authors", []))
            paper_url = top_paper.get("url", "No URL")

            return f"**{title}**\n {authors}\n {paper_url}"

    except Exception as e:
        return f"Error fetching citation: {e}"
