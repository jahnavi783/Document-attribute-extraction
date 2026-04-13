"""Generate sample files for demonstration."""
import io
import pandas as pd
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm
from openpyxl import Workbook


def create_sample_pdf_kv() -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Work Order Document", styles["Title"]))
    story.append(Spacer(1, 0.5*cm))

    data = [
        ["Workorder No", "WO-1023"],
        ["Customer", "Acme Corp"],
        ["Issue Date", "2024-01-15"],
        ["Due Date", "2024-02-01"],
        ["Emp ID", "EMP-4521"],
        ["Dept", "Engineering"],
        ["Total Cost", "₹45,200"],
        ["Tax", "₹8,136"],
        ["Grand Total", "₹53,336"],
        ["Current Status", "In Progress"],
        ["Notes", "Priority job, expedite delivery"],
        ["PO Number", "PO-2024-0087"],
    ]

    t = Table(data, colWidths=[7*cm, 10*cm])
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.lightgrey, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    doc.build(story)
    return buf.getvalue()


def create_sample_excel_tabular() -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Invoice Data"

    headers = ["Inv No", "Client Name", "Workorder No", "Qty", "Unit Cost",
               "Tax", "Grand Total", "Current Status", "Ph No"]
    ws.append(headers)

    rows = [
        ["INV-001", "TechCorp Ltd", "WO-1023", "5", "₹8,000", "₹7,200", "₹47,200", "Paid", "9876543210"],
        ["INV-002", "Global Inc", "WO-1024", "2", "₹15,000", "₹5,400", "₹35,400", "Pending", "8765432109"],
        ["INV-003", "Innovate Co", "WO-1025", "10", "₹2,500", "₹4,500", "₹29,500", "Paid", "7654321098"],
        ["INV-004", "BuildRight", "WO-1026", "3", "₹12,000", "₹6,480", "₹42,480", "In Review", "6543210987"],
    ]
    for row in rows:
        ws.append(row)

    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 18

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def create_sample_csv() -> bytes:
    df = pd.DataFrame({
        "Inv No": ["INV-101", "INV-102", "INV-103"],
        "Client Name": ["Alpha Corp", "Beta Ltd", "Gamma Inc"],
        "Workorder No": ["WO-2001", "WO-2002", "WO-2003"],
        "Issue Date": ["2024-03-01", "2024-03-05", "2024-03-10"],
        "Due Date": ["2024-04-01", "2024-04-05", "2024-04-10"],
        "Total Cost": ["₹25,000", "₹48,000", "₹12,500"],
        "Tax": ["₹4,500", "₹8,640", "₹2,250"],
        "Grand Total": ["₹29,500", "₹56,640", "₹14,750"],
        "Current Status": ["Paid", "Pending", "Paid"],
    })
    return df.to_csv(index=False).encode()


if __name__ == "__main__":
    out = Path("sample_data")
    out.mkdir(exist_ok=True)
    (out / "sample_workorder.pdf").write_bytes(create_sample_pdf_kv())
    (out / "sample_invoices.xlsx").write_bytes(create_sample_excel_tabular())
    (out / "sample_data.csv").write_bytes(create_sample_csv())
    print("Sample files created.")
