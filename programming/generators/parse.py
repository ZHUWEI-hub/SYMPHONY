"""Code parsing utilities for extracting code blocks from LLM outputs."""

import re
from typing import Optional


def parse_code_block(string: str, lang: str) -> Optional[str]:
    """Parse code block from markdown-style string.
    
    Args:
        string: String containing code block
        lang: Programming language ('python', 'rust', etc.)
        
    Returns:
        Extracted code or None if not found
    """
    code_pattern = fr"```{lang}\n(.*?)\n```"
    match = re.search(code_pattern, string, re.DOTALL)

    if match:
        return match.group(1)

    generic_code_pattern = r"```\n(.*?)\n```"
    match = re.search(generic_code_pattern, string, re.DOTALL)

    if match:
        return match.group(1)

    return parse_first_func(string, lang)

def parse_first_func(code: str, lang: str) -> Optional[str]:
    """Parse the first function from code string.
    
    Args:
        code: Source code string
        lang: Programming language ('python' or 'rust')
        
    Returns:
        First function definition or None if not found
    """
    code_lines = code.split("\n")

    if lang == "python":

        def_i = -1
        last_i = 0
        got_return = False
        for i, line in enumerate(code_lines):
            if line.startswith("def "):
                if def_i == -1:
                    def_i = i
                else:
                    break
            elif "return" in line and def_i != -1:
                got_return = True
            if line == "" and def_i != -1 and got_return:
                last_i = i
                break

        if last_i == 0:
            last_i = len(code_lines) - 1

        if def_i == -1:
            return None

        return "\n".join(code_lines[def_i:last_i+1]).rstrip("[/PYTHON]")

    elif lang == "rust":
        def_i = -1
        brace_count = 0
        in_function = False

        for i, line in enumerate(code_lines):
            stripped = line.strip()

            if def_i == -1:
                if re.match(r"^\s*fn\s+\w+.*->.*\{?", stripped):
                    def_i = i
                    brace_count += stripped.count('{')
                    brace_count -= stripped.count('}')
                    in_function = True
                    continue

            if in_function:
                brace_count += stripped.count('{')
                brace_count -= stripped.count('}')

                if brace_count <= 0:
                    return "\n".join(code_lines[def_i:i + 1]).rstrip("[/RUST]")

        if def_i != -1:
            return "\n".join(code_lines[def_i:]).rstrip("[/RUST]")

        func_pattern = r"(fn\s+\w+$.*?$\s*->\s*\w+\s*\{[\s\S]*?\n\})"
        match = re.search(func_pattern, code, re.DOTALL)
        return match.group(0).strip() if match else None

    else:
        raise ValueError(f"Unsupported language: {lang}")


def add_code_block(string: str, lang: str) -> str:
    """Wrap code in markdown code block.
    
    Args:
        string: Code string
        lang: Programming language
        
    Returns:
        Code wrapped in markdown code block
    """
    return f"```{lang}\n{string}\n```"


if __name__ == "__main__":
    CODE = """
aldaas
sub_parser = parser.add_subparsers().add_parser("frf
a")

def my_wonderful_func():
    def useless_helper():
        return 1
    if 1:
        return 1
    else:
        return (
            1,
            2,
        )

sadsadsa
2023-08-04dsa
dsa

def bleh():
    return aaa
"""
    print(parse_code_block(CODE, "python"))
    print("------------------------------------------------")
    CODE = """def total_match(lst1: List[str], lst2: List[str]) -> List[str]:
    \"\"\"
    Write a function that accepts two lists of strings and returns the list that has
    total number of chars in the all strings of the list less than the other list.
    
    if the two lists have the same number of chars, return the first list.
    
    Examples
    >>> total_match([], [])
    []
    >>> total_match(['hi', 'admin'], ['hI', 'Hi'])
    ['hI', 'Hi']
    >>> total_match(['hi', 'admin'], ['hi', 'hi', 'admin', 'project'])
    ['hi', 'admin']
    >>> total_match(['hi', 'admin'], ['hI', 'hi', 'hi'])
    ['hI', 'hi', 'hi']
    >>> total_match(['4'], ['1', '2', '3', '4', '5'])
    ['4']
    \"\"\"
    total_chars_lst1 = sum(len(word) for word in lst1)
    total_chars_lst2 = sum(len(word) for word in lst2)
    
    if total_chars_lst1 < total_chars_lst2:
        return lst1
    elif total_chars_lst1 > total_chars_lst2:
        return lst2
    else:
        return lst1
    """
    print(parse_code_block(CODE, "python"))
    print("------------------------------------------------")
    rust_code = """
    aaw qadqdqd
    /// 计算立方体体积
    fn voqwe 
    fn -? is
    fn volume_cube(l: isize) -> isize {
    
        l * l * l
    }
qwedqdqw
    #[test]
    fn test_volume() {
        assert_eq!(volume_cube(3), 27);
    }
    """
    print(parse_first_func(rust_code, "rust"))
