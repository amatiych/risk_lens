"""SEC EDGAR API client for 13F filing retrieval.

Handles SEC EDGAR API communication including:
- Rate limiting (SEC requires max 10 requests/second)
- Required User-Agent headers
- 13F filing search and download
"""

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

import requests


SEC_USER_AGENT = "Risk-Lens aleksey@purpleswansoftware.com"
SEC_BASE_URL = "https://data.sec.gov"


@dataclass
class SECFiler:
    """SEC filer information."""

    cik: str
    name: str
    entity_type: Optional[str] = None


@dataclass
class Filing13F:
    """13F filing metadata."""

    accession_number: str
    filing_date: str
    report_date: str
    form_type: str
    primary_document: Optional[str] = None


@dataclass
class Holding13F:
    """Individual holding from 13F Information Table."""

    cusip: str
    name: str
    title_of_class: str
    value: int  # In thousands of dollars
    shares: int
    share_type: str  # "SH" for shares, "PRN" for principal amount
    investment_discretion: str
    voting_authority_sole: int
    voting_authority_shared: int
    voting_authority_none: int


class SECEdgarClient:
    """Client for SEC EDGAR API.

    Implements rate limiting and proper User-Agent headers as required by SEC.
    """

    def __init__(self, user_agent: str = SEC_USER_AGENT):
        """Initialize the SEC EDGAR client.

        Args:
            user_agent: User-Agent string for SEC compliance.
        """
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        })
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce SEC rate limit of 10 requests/second."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.1:  # 100ms between requests
            time.sleep(0.1 - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, **kwargs: Any) -> requests.Response:
        """Make a rate-limited GET request.

        Args:
            url: URL to request.
            **kwargs: Additional arguments for requests.get.

        Returns:
            Response object.

        Raises:
            requests.HTTPError: If the request fails.
        """
        self._rate_limit()
        response = self.session.get(url, **kwargs)
        response.raise_for_status()
        return response

    def search_filer(self, company_name: str) -> List[SECFiler]:
        """Search for institutional investors by name.

        Uses SEC EDGAR full-text search API to find 13F filers.

        Args:
            company_name: Name or partial name of the institutional investor.

        Returns:
            List of matching SECFiler objects.
        """
        # Use SEC's full-text search API for 13F filings
        search_url = "https://efts.sec.gov/LATEST/search-index"

        try:
            self._rate_limit()
            # Search for 13F filings with the company name
            params = {
                "q": f'"{company_name}"',
                "dateRange": "custom",
                "forms": "13F-HR",
                "startdt": "2020-01-01",
                "enddt": "2030-01-01",
            }
            response = self.session.get(search_url, params=params)

            # If full-text search doesn't work, try the company search endpoint
            if response.status_code != 200:
                return self._search_via_company_endpoint(company_name)

            data = response.json()
            hits = data.get("hits", {}).get("hits", [])

            # Extract unique filers from search results
            seen_ciks = set()
            matches = []

            for hit in hits[:50]:  # Check first 50 results
                source = hit.get("_source", {})
                cik = source.get("ciks", [""])[0] if source.get("ciks") else ""
                name = (
                    source.get("display_names", [""])[0]
                    if source.get("display_names")
                    else ""
                )

                if cik and cik not in seen_ciks:
                    seen_ciks.add(cik)
                    matches.append(
                        SECFiler(
                            cik=str(cik).zfill(10),
                            name=name,
                            entity_type="13F Filer",
                        )
                    )

            if matches:
                return matches[:20]

            # Fallback to company search endpoint
            return self._search_via_company_endpoint(company_name)

        except (requests.RequestException, KeyError, ValueError):
            return self._search_via_company_endpoint(company_name)

    def _search_via_company_endpoint(
        self, company_name: str
    ) -> List[SECFiler]:
        """Search using SEC company search endpoint.

        Args:
            company_name: Name to search for.

        Returns:
            List of matching SECFiler objects.
        """
        try:
            # Use SEC company search
            search_url = (
                "https://www.sec.gov/cgi-bin/browse-edgar"
                f"?company={company_name.replace(' ', '+')}"
                "&CIK=&type=13F-HR&owner=include&count=40&action=getcompany"
            )

            self._rate_limit()
            # Get HTML and parse for CIKs
            response = self.session.get(search_url)
            response.raise_for_status()

            # Simple regex to extract CIKs and names from the HTML
            # Pattern: /cgi-bin/browse-edgar?action=getcompany&CIK=NNNNNNNNNN
            import re

            cik_pattern = r'CIK=(\d+)[^>]*>([^<]+)</a>'
            matches_raw = re.findall(cik_pattern, response.text)

            matches = []
            seen_ciks = set()

            for cik, name in matches_raw:
                if cik not in seen_ciks:
                    seen_ciks.add(cik)
                    matches.append(
                        SECFiler(
                            cik=str(cik).zfill(10),
                            name=name.strip(),
                            entity_type=None,
                        )
                    )

            return matches[:20]

        except requests.RequestException:
            # Final fallback: direct CIK lookup if name looks like a number
            if company_name.isdigit():
                return self._lookup_by_cik(company_name)
            return []

    def _lookup_by_cik(self, cik: str) -> List[SECFiler]:
        """Lookup filer by CIK directly.

        Args:
            cik: Central Index Key.

        Returns:
            List containing the filer if found.
        """
        cik_padded = cik.zfill(10)
        url = f"{SEC_BASE_URL}/submissions/CIK{cik_padded}.json"

        try:
            response = self._get(url)
            data = response.json()
            return [
                SECFiler(
                    cik=cik_padded,
                    name=data.get("name", ""),
                    entity_type=data.get("entityType"),
                )
            ]
        except requests.RequestException:
            return []

    def get_13f_filings(self, cik: str, limit: int = 10) -> List[Filing13F]:
        """Get recent 13F-HR filings for a CIK.

        Args:
            cik: Central Index Key (with or without leading zeros).
            limit: Maximum number of filings to return.

        Returns:
            List of Filing13F objects, most recent first.
        """
        cik_padded = cik.zfill(10)
        url = f"{SEC_BASE_URL}/submissions/CIK{cik_padded}.json"

        response = self._get(url)
        data = response.json()

        filings = []
        recent_filings = data.get("filings", {}).get("recent", {})

        forms = recent_filings.get("form", [])
        accession_numbers = recent_filings.get("accessionNumber", [])
        filing_dates = recent_filings.get("filingDate", [])
        report_dates = recent_filings.get("reportDate", [])
        primary_docs = recent_filings.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form in ("13F-HR", "13F-HR/A"):  # Include amendments
                if len(filings) >= limit:
                    break

                filings.append(
                    Filing13F(
                        accession_number=accession_numbers[i],
                        filing_date=filing_dates[i],
                        report_date=report_dates[i] if i < len(report_dates) else "",
                        form_type=form,
                        primary_document=primary_docs[i]
                        if i < len(primary_docs)
                        else None,
                    )
                )

        return filings

    def get_13f_holdings(
        self, cik: str, accession_number: str
    ) -> List[Holding13F]:
        """Download and parse 13F Information Table.

        Args:
            cik: Central Index Key.
            accession_number: Accession number of the filing.

        Returns:
            List of Holding13F objects from the Information Table.
        """
        cik_padded = cik.zfill(10)
        accession_clean = accession_number.replace("-", "")

        # First, get the filing index to find the information table file
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik_padded)}/{accession_clean}/index.json"
        )

        try:
            response = self._get(index_url)
            index_data = response.json()

            # Find the information table XML file
            # The holdings are typically in a separate XML file, not the primary doc
            info_table_file = None
            xml_files = []

            for item in index_data.get("directory", {}).get("item", []):
                name = item.get("name", "").lower()
                if name.endswith(".xml"):
                    xml_files.append(item.get("name"))
                    # Check for common naming patterns for holdings
                    if "infotable" in name:
                        info_table_file = item.get("name")
                        break
                    elif "holding" in name:
                        info_table_file = item.get("name")
                        break
                    elif "13f" in name and "primary" not in name:
                        info_table_file = item.get("name")

            # If no specific holdings file found, try the non-primary XML files
            if not info_table_file and xml_files:
                for xml_file in xml_files:
                    if "primary" not in xml_file.lower():
                        info_table_file = xml_file
                        break
                # Last resort: use any XML file
                if not info_table_file:
                    info_table_file = xml_files[0]

            if not info_table_file:
                raise ValueError(
                    f"Could not find information table XML in filing {accession_number}"
                )

            # Download and parse the information table
            xml_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik_padded)}/{accession_clean}/{info_table_file}"
            )

            self._rate_limit()
            xml_response = self.session.get(xml_url)
            xml_response.raise_for_status()

            return self._parse_info_table_xml(xml_response.text)

        except requests.RequestException as e:
            raise ValueError(f"Failed to retrieve 13F holdings: {e}") from e

    def _parse_info_table_xml(self, xml_content: str) -> List[Holding13F]:
        """Parse 13F Information Table XML.

        Args:
            xml_content: Raw XML content of the information table.

        Returns:
            List of Holding13F objects.
        """
        # Handle namespace variations in 13F XML
        # More aggressive namespace removal for compatibility
        xml_clean = xml_content

        # Remove XML declaration if present
        xml_clean = re.sub(r'<\?xml[^?]*\?>', '', xml_clean)

        # Remove all namespace declarations
        xml_clean = re.sub(r'\sxmlns[^=]*="[^"]*"', '', xml_clean)

        # Remove namespace prefixes from tags
        xml_clean = re.sub(r'<([a-zA-Z0-9]+):', '<', xml_clean)
        xml_clean = re.sub(r'</([a-zA-Z0-9]+):', '</', xml_clean)

        # Also handle ns1:, n1:, etc.
        xml_clean = re.sub(r'<ns\d+:', '<', xml_clean)
        xml_clean = re.sub(r'</ns\d+:', '</', xml_clean)
        xml_clean = re.sub(r'<n\d+:', '<', xml_clean)
        xml_clean = re.sub(r'</n\d+:', '</', xml_clean)

        # Remove any remaining namespace attributes
        xml_clean = re.sub(r'\s[a-zA-Z0-9]+:[a-zA-Z0-9]+="[^"]*"', '', xml_clean)

        try:
            root = ElementTree.fromstring(xml_clean)
        except ElementTree.ParseError:
            # Try with encoding fix
            try:
                xml_clean = xml_content.encode("utf-8", errors="ignore").decode(
                    "utf-8"
                )
                xml_clean = re.sub(r'\sxmlns[^=]*="[^"]*"', '', xml_clean)
                xml_clean = re.sub(r'<([a-zA-Z0-9]+):', '<', xml_clean)
                xml_clean = re.sub(r'</([a-zA-Z0-9]+):', '</', xml_clean)
                root = ElementTree.fromstring(xml_clean)
            except ElementTree.ParseError as e:
                raise ValueError(f"Failed to parse 13F XML: {e}") from e

        holdings = []

        # Find all infoTable entries (case-insensitive search)
        for info_table in root.iter():
            if info_table.tag.lower().endswith("infotable"):
                holding = self._parse_holding_entry(info_table)
                if holding:
                    holdings.append(holding)

        return holdings

    def _parse_holding_entry(
        self, entry: ElementTree.Element
    ) -> Optional[Holding13F]:
        """Parse a single holding entry from the XML.

        Args:
            entry: XML element for the holding.

        Returns:
            Holding13F object or None if parsing fails.
        """

        def get_text(element: ElementTree.Element, tag: str) -> str:
            """Get text from child element, case-insensitive."""
            for child in element:
                if child.tag.lower().endswith(tag.lower()):
                    return child.text or ""
                # Check nested elements
                result = get_text(child, tag)
                if result:
                    return result
            return ""

        def get_int(element: ElementTree.Element, tag: str) -> int:
            """Get integer from child element."""
            text = get_text(element, tag)
            try:
                return int(text.replace(",", ""))
            except (ValueError, AttributeError):
                return 0

        try:
            cusip = get_text(entry, "cusip")
            if not cusip:
                return None

            # Clean CUSIP (should be 9 characters)
            cusip = cusip.strip().upper()

            return Holding13F(
                cusip=cusip,
                name=get_text(entry, "nameOfIssuer"),
                title_of_class=get_text(entry, "titleOfClass"),
                value=get_int(entry, "value"),
                shares=get_int(entry, "sshPrnamt"),
                share_type=get_text(entry, "sshPrnamtType") or "SH",
                investment_discretion=get_text(entry, "investmentDiscretion"),
                voting_authority_sole=get_int(entry, "Sole"),
                voting_authority_shared=get_int(entry, "Shared"),
                voting_authority_none=get_int(entry, "None"),
            )
        except Exception:
            return None
