"""Wikipedia environment for interactive question answering."""

import ast
import json
import time

import gym
import requests
import certifi
from bs4 import BeautifulSoup
from urllib3.util.ssl_ import create_urllib3_context


def clean_str(p):
    """Clean up unicode escape sequences in strings.
    
    Args:
        p: Input string
        
    Returns:
        Cleaned string
    """
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except UnicodeDecodeError:
        return p


class textSpace(gym.spaces.Space):
    """Custom gym space for text observations."""
    
    def contains(self, x) -> bool:
        """Check if x is a valid text observation."""
        return isinstance(x, str)


class TLSAdapter(requests.adapters.HTTPAdapter):
    """HTTP adapter with custom TLS configuration."""
    
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.options |= 0x00000400  # OP_NO_SSLv3
        context.options |= 0x00000800  # OP_NO_TLSv1
        context.options |= 0x00001000  # OP_NO_TLSv1_1
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)


class WikiEnv(gym.Env):
    """Wikipedia search environment for question answering."""

    def __init__(self):
        """Initialize the Wikipedia environment."""
        super().__init__()
        self.page = None
        self.obs = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        self.observation_space = self.action_space = textSpace()
        self.search_time = 0
        self.num_searches = 0
    
  def _get_obs(self):
    return self.obs

  def _get_info(self):
    return {"steps": self.steps, "answer": self.answer}

  def reset(self, seed=None, return_info=False, options=None):
    """Reset the environment to initial state.
    
    Args:
        seed: Random seed (unused)
        return_info: Whether to return info dict
        options: Additional options (unused)
        
    Returns:
        Initial observation, and optionally info dict
    """
    self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                "finish[].\n")
    self.page = None
    self.lookup_keyword = None
    self.lookup_list = None
    self.lookup_cnt = None
    self.steps = 0
    self.answer = None
    observation = self._get_obs()
    info = self._get_info()
    return (observation, info) if return_info else observation

  def construct_lookup_list(self, keyword):
    """Find all sentences containing the keyword in the current page.
    
    Args:
        keyword: The keyword to search for
        
    Returns:
        List of sentences containing the keyword
    """
    if self.page is None:
      return []
    paragraphs = self.page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]

    parts = sentences
    parts = [p for p in parts if keyword.lower() in p.lower()]
    return parts

  @staticmethod
  def get_page_obs(page):
    """Extract the first few sentences from a Wikipedia page.
    
    Args:
        page: The full page content
        
    Returns:
        String with the first 5 sentences
    """
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    return ' '.join(sentences[:5])

  def search_step(self, entity):
    """Search for an entity on Wikipedia.
    
    Args:
        entity: The entity name to search for
    """
    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    old_time = time.time()

    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
      total=7,
      backoff_factor=0.3,
      status_forcelist=[500, 502, 503, 504],
      allowed_methods=frozenset(['GET', 'POST'])
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=100,
        pool_maxsize=100
    )

    session.mount('https://', adapter)
    session.mount('http://', adapter)

    response = session.get(
      search_url,
      stream=True,
      verify=certifi.where(),
      timeout=15,
      headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
      }
    )

    response_text = ""
    try:
      for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
        if chunk:
          response_text += chunk
    except requests.exceptions.ChunkedEncodingError as e:
      print(f"Chunked encoding error: {e}")
    finally:
      response.close()

    self.search_time += time.time() - old_time
    self.num_searches += 1
    soup = BeautifulSoup(response_text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    
    if result_divs:
      self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
      self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
    else:
      page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
      if any("may refer to:" in p for p in page):
        self.search_step("[" + entity + "]")
      else:
        self.page = ""
        for p in page:
          if len(p.split(" ")) > 2:
            self.page += clean_str(p)
            if not p.endswith("\n"):
              self.page += "\n"
        self.obs = self.get_page_obs(self.page)
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
  
  def step(self, action):
    """Execute an action in the environment.
    
    Args:
        action: Action string (search[entity], lookup[keyword], or finish[answer])
        
    Returns:
        Tuple of (observation, reward, done, info)
    """
    reward = 0
    done = False
    action = action.strip()
    
    if action.startswith("search[") and action.endswith("]"):
      entity = action[len("search["):-1]
      self.search_step(entity)
    elif action.startswith("lookup[") and action.endswith("]"):
      keyword = action[len("lookup["):-1]
      if self.lookup_keyword != keyword:
        self.lookup_keyword = keyword
        self.lookup_list = self.construct_lookup_list(keyword)
        self.lookup_cnt = 0
      if self.lookup_cnt >= len(self.lookup_list):
        self.obs = "No more results.\n"
      else:
        self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
        self.lookup_cnt += 1
    elif action.startswith("finish[") and action.endswith("]"):
      answer = action[len("finish["):-1]
      self.answer = answer
      done = True
      self.obs = f"Episode finished, reward = {reward}\n"
    elif action.startswith("think[") and action.endswith("]"):
      self.obs = "Nice thought."
    else:
      self.obs = "Invalid action: {}".format(action)

    self.steps += 1
    return self.obs, reward, done, self._get_info()
  
  def get_time_info(self):
    """Get timing information for search operations.
    
    Returns:
        Dictionary with call_speed, call_time, and num_calls
    """
    speed = self.search_time / self.num_searches if self.num_searches else 0
    return {
        "call_speed": speed,
        "call_time": self.search_time,
        "num_calls": self.num_searches,
    }
