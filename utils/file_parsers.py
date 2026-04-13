"""
File Parsers - Extract attribute-value pairs from PDF, Excel, and CSV files.

Behavior for this project:
- Prefer clean key-value extraction for normalization output.
- Even when source looks tabular, if rows behave like form rows, extract them as
  attribute/value pairs.
"""

import re
import io
import pdfplumber
import pandas as pd
from typing import Union

_SKIP_PATTERNS = re.compile(
    r"^(field\s*label|attribute|key|column|header|name|label|field|value|#|no\.?)$",
    re.IGNORECASE,
)


def _clean_cell(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


def _is_header_row(attr: str, val: str) -> bool:
    attr = (attr or '').strip()
    val = (val or '').strip()
    if not attr:
        return True
    # common header rows
    if _SKIP_PATTERNS.match(attr):
        return True
    if attr.lower() in {'field label', 'attribute', 'name'} and val.lower() in {'value', 'values'}:
        return True
    return False


def _row_to_kv(row) -> tuple[str, str]:
    """Return (attr, value) using first non-empty cell as attribute and next non-empty cell as value."""
    cells = [_clean_cell(c) for c in row if _clean_cell(c)]
    if len(cells) < 2:
        return "", ""
    attr = cells[0]
    value = cells[1]
    return attr, value


def parse_pdf(file: Union[str, io.BytesIO]) -> tuple[list[dict], str]:
    records = []
    doc_type = "keyvalue"
    with pdfplumber.open(file) as pdf:
        all_text = []
        seen = set()
        for page in pdf.pages:
            tables = page.extract_tables() or []
            page_had_table = False
            for table in tables:
                if not table:
                    continue
                page_had_table = True
                clean_rows = [row for row in table if row and any(_clean_cell(c) for c in row)]
                for row in clean_rows:
                    attr, val = _row_to_kv(row)
                    if not attr or _is_header_row(attr, val):
                        continue
                    # don't allow pure number as attribute
                    if re.fullmatch(r"[\d\s.,]+", attr):
                        continue
                    key = (attr.lower(), val)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append({"attribute": attr, "value": val})
            if not page_had_table:
                all_text.append(page.extract_text() or "")
        # fallback plain text only when no records from tables
        if not records:
            for line in "\n".join(all_text).splitlines():
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                m = re.match(r"^([A-Za-z][^:=\-]{1,80}?)[\s]*[:=\-]+[\s]*(.*)$", line)
                if m:
                    attr = m.group(1).strip()
                    val = m.group(2).strip()
                    if _is_header_row(attr, val):
                        continue
                    records.append({"attribute": attr, "value": val})
    return records, doc_type


def parse_pdf_as_dataframes(file: Union[str, io.BytesIO]) -> list[pd.DataFrame]:
    dfs = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                if not table:
                    continue
                rows = [[_clean_cell(c) for c in row] for row in table if row and any(_clean_cell(c) for c in row)]
                if len(rows) >= 2:
                    header = [c or f"Col{i}" for i, c in enumerate(rows[0])]
                    dfs.append(pd.DataFrame(rows[1:], columns=header))
    return dfs


def parse_excel(file: Union[str, io.BytesIO]) -> tuple[dict[str, pd.DataFrame], str]:
    xl = pd.read_excel(file, sheet_name=None, header=None, dtype=str)
    sheets = {}
    for sheet_name, df in xl.items():
        df = df.dropna(how='all').dropna(axis=1, how='all').fillna('')
        df = df.apply(lambda col: col.map(lambda x: str(x).strip()))
        df = df[df.apply(lambda row: any(_clean_cell(v) for v in row), axis=1)].reset_index(drop=True)
        if not df.empty:
            sheets[sheet_name] = df
    # For this project, normalize into keyvalue output even if source was table
    return sheets, 'keyvalue'


def extract_kv_from_excel_sheet(df: pd.DataFrame) -> list[dict]:
    records = []
    seen = set()
    for _, row in df.iterrows():
        attr, val = _row_to_kv(list(row))
        if not attr or _is_header_row(attr, val):
            continue
        key = (attr.lower(), val)
        if key in seen:
            continue
        seen.add(key)
        records.append({"attribute": attr, "value": val})
    return records


def extract_tabular_from_excel_sheet(df: pd.DataFrame) -> pd.DataFrame:
    # kept for compatibility; not used in the final key-value flow
    if df.empty:
        return df
    headers = [(_clean_cell(v) or f"Col{i}") for i, v in enumerate(df.iloc[0].tolist())]
    new_df = df.iloc[1:].copy()
    new_df.columns = headers
    new_df = new_df[new_df.apply(lambda row: any(_clean_cell(v) for v in row), axis=1)]
    return new_df.reset_index(drop=True)


def parse_csv(file: Union[str, io.BytesIO]) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(file, header=None, dtype=str).fillna('')
    df = df[df.apply(lambda row: any(_clean_cell(v) for v in row), axis=1)].reset_index(drop=True)
    return df, 'keyvalue'