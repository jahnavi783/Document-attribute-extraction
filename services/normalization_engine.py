"""
Normalization Engine - parse -> match -> normalize -> write
Final output for this project is always a clean 2-column normalized document:
Attribute | Value
"""

import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import pandas as pd

from services.attribute_matcher import AttributeMatcher, MatchResult, DEFAULT_FUZZY_THRESHOLD, DEFAULT_SEMANTIC_THRESHOLD
from utils.file_parsers import parse_pdf, parse_excel, extract_kv_from_excel_sheet, parse_csv
from utils.file_writers import write_pdf_keyvalue, write_excel_keyvalue, write_csv


@dataclass
class NormalizationReport:
    input_format: str
    doc_type: str
    total_attributes: int
    matched: int
    unmatched: int
    match_details: list[MatchResult]
    output_bytes: bytes
    output_ext: str
    normalized_records: list[dict] = field(default_factory=list)


class NormalizationEngine:
    def __init__(self, master_path: str):
        self.matcher = AttributeMatcher(master_path)

    def _norm_attr(self, attr: str, fuzzy_threshold: float, semantic_threshold: float):
        result = self.matcher.match(attr, fuzzy_threshold=fuzzy_threshold, semantic_threshold=semantic_threshold)
        return result.canonical_attr, result

    def _normalize_records(self, records: list[dict], fuzzy_threshold: float, semantic_threshold: float):
        normalized = []
        match_results = []
        for rec in records:
            canonical, result = self._norm_attr(str(rec.get('attribute','')), fuzzy_threshold, semantic_threshold)
            match_results.append(result)
            normalized.append({"attribute": canonical, "value": rec.get('value','')})
        return normalized, match_results

    def process_pdf(self, file: io.BytesIO, filename: str, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD, semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD) -> NormalizationReport:
        file.seek(0)
        records, doc_type = parse_pdf(file)
        norm_records, all_results = self._normalize_records(records, fuzzy_threshold, semantic_threshold)
        out = write_pdf_keyvalue(norm_records, title=filename)
        matched = sum(1 for r in all_results if r.match_type != 'unmatched')
        return NormalizationReport('pdf', doc_type, len(all_results), matched, len(all_results)-matched, all_results, out, 'pdf', norm_records)

    def process_excel(self, file: io.BytesIO, filename: str, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD, semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD) -> NormalizationReport:
        file.seek(0)
        sheets, doc_type = parse_excel(file)
        all_records = []
        all_results = []
        for _, df in sheets.items():
            records = extract_kv_from_excel_sheet(df)
            norm_records, results = self._normalize_records(records, fuzzy_threshold, semantic_threshold)
            all_records.extend(norm_records)
            all_results.extend(results)
        out = write_excel_keyvalue(all_records)
        matched = sum(1 for r in all_results if r.match_type != 'unmatched')
        return NormalizationReport('excel', doc_type, len(all_results), matched, len(all_results)-matched, all_results, out, 'xlsx', all_records)

    def process_csv(self, file: io.BytesIO, filename: str, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD, semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD) -> NormalizationReport:
        file.seek(0)
        df, doc_type = parse_csv(file)
        records = []
        for _, row in df.iterrows():
            cells = [str(v).strip() for v in row.tolist() if str(v).strip() and str(v).strip().lower() not in ('nan','none')]
            if len(cells) >= 2 and cells[0].lower() not in ('field label','attribute','name'):
                records.append({'attribute': cells[0], 'value': cells[1]})
        norm_records, all_results = self._normalize_records(records, fuzzy_threshold, semantic_threshold)
        out_df = pd.DataFrame([{'Attribute': r['attribute'], 'Value': r['value']} for r in norm_records])
        out = write_csv(out_df)
        matched = sum(1 for r in all_results if r.match_type != 'unmatched')
        return NormalizationReport('csv', doc_type, len(all_results), matched, len(all_results)-matched, all_results, out, 'csv', norm_records)

    def process(self, file: io.BytesIO, filename: str, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD, semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD) -> NormalizationReport:
        ext = Path(filename).suffix.lower()
        if ext == '.pdf':
            return self.process_pdf(file, filename, fuzzy_threshold, semantic_threshold)
        elif ext in ('.xlsx','.xls'):
            return self.process_excel(file, filename, fuzzy_threshold, semantic_threshold)
        elif ext == '.csv':
            return self.process_csv(file, filename, fuzzy_threshold, semantic_threshold)
        raise ValueError(f'Unsupported file type: {ext}')
