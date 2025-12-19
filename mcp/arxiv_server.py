#!/usr/bin/env python3
"""
MCP server for fetching arXiv papers as plain text.

Input: arXiv ID (e.g., "2502.16681" or "2502.16681v1")
Output: Full paper text extracted from HTML

Usage:
    Add to ~/.claude/settings.json:
    {
      "mcpServers": {
        "arxiv": {
          "command": "python",
          "args": ["/path/to/arxiv_server.py"]
        }
      }
    }
"""

import re
import time
import requests
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arxiv")

# Rate limiting: arXiv asks for 1 request per 3 seconds
_last_request_time = 0

def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < 3:
        time.sleep(3 - elapsed)
    _last_request_time = time.time()

def _clean_arxiv_id(arxiv_id: str) -> str:
    """Extract arxiv ID from various input formats."""
    # Handle full URLs
    if "arxiv.org" in arxiv_id:
        match = re.search(r'(\d{4}\.\d{4,5})(v\d+)?', arxiv_id)
        if match:
            return match.group(0)
    # Handle bare IDs
    match = re.match(r'^(\d{4}\.\d{4,5})(v\d+)?$', arxiv_id.strip())
    if match:
        return match.group(0)
    return arxiv_id.strip()

def _extract_text_from_html(html: str) -> str:
    """Extract clean text from arXiv HTML."""
    soup = BeautifulSoup(html, 'html.parser')

    # Remove script, style, nav elements
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()

    # Try to find main article content
    article = soup.find('article') or soup.find('main') or soup.find('div', class_='ltx_page_content')

    if article:
        content = article
    else:
        content = soup.body or soup

    # Extract text with some structure preservation
    lines = []

    for element in content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'figcaption', 'td', 'th']):
        text = element.get_text(separator=' ', strip=True)
        if not text:
            continue

        # Add markdown-style headers
        if element.name == 'h1':
            lines.append(f"\n# {text}\n")
        elif element.name == 'h2':
            lines.append(f"\n## {text}\n")
        elif element.name == 'h3':
            lines.append(f"\n### {text}\n")
        elif element.name == 'h4':
            lines.append(f"\n#### {text}\n")
        elif element.name == 'li':
            lines.append(f"- {text}")
        else:
            lines.append(text)

    return '\n'.join(lines)

@mcp.tool()
def fetch_arxiv_paper(arxiv_id: str) -> str:
    """
    Fetch the full text of an arXiv paper from its HTML version.

    Args:
        arxiv_id: arXiv paper ID (e.g., "2502.16681", "2502.16681v1", or full URL)

    Returns:
        Full paper text in markdown-ish format
    """
    _rate_limit()

    clean_id = _clean_arxiv_id(arxiv_id)
    url = f"https://arxiv.org/html/{clean_id}"

    headers = {
        'User-Agent': 'arxiv-mcp-server/1.0 (research tool; respects rate limits)'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return f"Error: Paper {clean_id} not found. Note: Not all arXiv papers have HTML versions (only newer papers do)."
        return f"Error fetching paper: {e}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching paper: {e}"

    text = _extract_text_from_html(response.text)

    if len(text) < 500:
        return f"Error: Could not extract meaningful content from {url}. The paper may not have an HTML version available."

    return f"# arXiv:{clean_id}\nSource: {url}\n\n{text}"

@mcp.tool()
def fetch_arxiv_abstract(arxiv_id: str) -> str:
    """
    Fetch just the abstract and metadata of an arXiv paper (faster, works for all papers).

    Args:
        arxiv_id: arXiv paper ID (e.g., "2502.16681")

    Returns:
        Paper title, authors, and abstract
    """
    _rate_limit()

    clean_id = _clean_arxiv_id(arxiv_id)
    url = f"https://arxiv.org/abs/{clean_id}"

    headers = {
        'User-Agent': 'arxiv-mcp-server/1.0 (research tool; respects rate limits)'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error fetching paper: {e}"

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract title
    title_elem = soup.find('h1', class_='title')
    title = title_elem.get_text(strip=True).replace('Title:', '').strip() if title_elem else "Unknown"

    # Extract authors
    authors_elem = soup.find('div', class_='authors')
    authors = authors_elem.get_text(strip=True).replace('Authors:', '').strip() if authors_elem else "Unknown"

    # Extract abstract
    abstract_elem = soup.find('blockquote', class_='abstract')
    abstract = abstract_elem.get_text(strip=True).replace('Abstract:', '').strip() if abstract_elem else "No abstract"

    return f"# {title}\n\n**Authors:** {authors}\n\n**Abstract:** {abstract}\n\n**URL:** {url}"

if __name__ == "__main__":
    mcp.run()
